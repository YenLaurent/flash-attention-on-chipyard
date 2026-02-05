package softmax

import chisel3._
import chisel3.util._
import _root_.circt.stage.ChiselStage
import top.ForwardSlice


class S1toS2Bundle extends Bundle {
  val original_x = new BFloat16()
  val y = new BFloat16()
  val is_special = Bool()
  val special_result = new BFloat16()
}

class S2toS3Bundle extends Bundle {
  val original_x = new BFloat16() 
  val y = new BFloat16()
  val is_special = Bool()
  val special_result = new BFloat16()
}


class S3toS4Bundle extends Bundle {
  val I = SInt(16.W)
  val pow2_F = new BFloat16()
  val is_special = Bool()
  val special_result = new BFloat16()
}

object BF16ExpUnit {
  val BIAS = 127
  val MANTISSA_WIDTH = 7
  val EXPONENT_WIDTH = 8

  val LOG2E_BF16 = "h3FB9".U(16.W)
  val ONE_BF16   = "h3F80".U(16.W)

  def nan(): BFloat16 = {
    val w = Wire(new BFloat16)
    w.sign := false.B
    w.exponent := ((1 << EXPONENT_WIDTH) - 1).U
    //w.mantissa := 1.U
    w.mantissa := "b1000000".U
    w
  }

  def inf(sign: Bool): BFloat16 = {
    val w = Wire(new BFloat16)
    w.sign := sign
    w.exponent := ((1 << EXPONENT_WIDTH) - 1).U
    w.mantissa := 0.U
    w
  }

  def zero(sign: Bool): BFloat16 = {
    val w = Wire(new BFloat16)
    w.sign := sign
    w.exponent := 0.U
    w.mantissa := 0.U
    w
  }
  
  def generateLUT(): Seq[UInt] = {
    (0 until 128).map { i =>
      val f_value = i.toDouble / 128.toDouble
      val pow2_f = math.pow(2.0, f_value)
      val floatBits = java.lang.Float.floatToIntBits(pow2_f.toFloat)
      ((floatBits >>> 16) & 0xFFFF).U(16.W)
    }
  }
}


class BF16ExpUnit extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new BFloat16()))
    val out = Decoupled(new BFloat16())
  })

  import BF16ExpUnit._

  val lut_2_pow_F = VecInit(generateLUT())
  val mul_log2e = Module(new BF16Mul())
  val sub_unit_A = Module(new BF16Adder())
  val sub_unit_B = Module(new BF16Adder())
  

  val s1_q = Module(new Queue(new BFloat16(), 1, pipe = true))
  
  val s2_q = Module(new Queue(new S1toS2Bundle(), 1, pipe = true))
  val s3_q = Module(new Queue(new S2toS3Bundle(), 1, pipe = true))
  val s4_q = Module(new Queue(new S3toS4Bundle(), 1, pipe = true))
  val s5_q = Module(new Queue(new BFloat16(), 1, pipe = true))

  s1_q.io.enq <> io.in
  io.out <> s5_q.io.deq
  


  val s1_bits = s1_q.io.deq.bits
  val is_zero = s1_bits.exponent === 0.U && s1_bits.mantissa === 0.U
  val is_denormal = s1_bits.exponent === 0.U && s1_bits.mantissa =/= 0.U
  val is_inf = s1_bits.exponent === 255.U && s1_bits.mantissa === 0.U
  val is_nan = s1_bits.exponent === 255.U && s1_bits.mantissa =/= 0.U
  val is_special = is_zero || is_denormal || is_inf || is_nan

  val special_result = Wire(new BFloat16())
  when(is_zero || is_denormal) { special_result := BFloat16(ONE_BF16) }
  .elsewhen(is_inf && !s1_bits.sign) { special_result := inf(false.B) }
  .elsewhen(is_inf && s1_bits.sign) { special_result := zero(false.B) }
  .otherwise { special_result := nan() }

  val input_for_mul = Mux(is_denormal, zero(s1_bits.sign), s1_bits)
  mul_log2e.io.a := input_for_mul
  mul_log2e.io.b := BFloat16(LOG2E_BF16)

  s2_q.io.enq.valid := s1_q.io.deq.valid
  s1_q.io.deq.ready := s2_q.io.enq.ready

  s2_q.io.enq.bits.original_x := s1_bits 
  s2_q.io.enq.bits.y := mul_log2e.io.result
  s2_q.io.enq.bits.is_special := is_special
  s2_q.io.enq.bits.special_result := special_result


  s3_q.io.enq <> s2_q.io.deq

  val s3_bits = s3_q.io.deq.bits
  val y = s3_bits.y
  val y_exp_unbiased = y.exponent.zext - BIAS.S

  val y_trunc_bf16 = Wire(new BFloat16())
  // ... (y_trunc_bf16 logic)
  when(y_exp_unbiased < 0.S) {
    y_trunc_bf16 := zero(y.sign)
  }.elsewhen(y_exp_unbiased >= MANTISSA_WIDTH.S) {
    y_trunc_bf16 := y
  }.otherwise {
    val num_frac_bits = (MANTISSA_WIDTH.S - y_exp_unbiased).asUInt
    val mask = ~(((1.U << num_frac_bits) - 1.U)(MANTISSA_WIDTH - 1, 0))
    y_trunc_bf16.sign := y.sign
    y_trunc_bf16.exponent := y.exponent
    y_trunc_bf16.mantissa := y.mantissa & mask
  }
  val needs_floor_correction = y.sign && (y.asBits() =/= y_trunc_bf16.asBits())
  sub_unit_A.io.a := y_trunc_bf16
  sub_unit_A.io.b := BFloat16(ONE_BF16)
  sub_unit_A.io.sub := true.B
  val y_floor_bf16 = Mux(needs_floor_correction, sub_unit_A.io.result, y_trunc_bf16)
  
  val I = Wire(SInt(16.W))
  // ... (I from y_floor_bf16 logic)
  val floor_exp_unbiased = y_floor_bf16.exponent.zext - BIAS.S
  val floor_sig = Cat(1.U(1.W), y_floor_bf16.mantissa)
  when(y_floor_bf16.exponent === 0.U) {
      I := 0.S
  } .elsewhen (floor_exp_unbiased >= MANTISSA_WIDTH.S) {
      val shift_left = floor_exp_unbiased.asUInt - MANTISSA_WIDTH.U
      val magnitude = (floor_sig << shift_left).asSInt
      I := Mux(y_floor_bf16.sign, -magnitude, magnitude)
  } .otherwise {
      //val shift_right = MANTISSA_WIDTH.U - floor_exp_unbiased.asUInt
      val shift_right = (MANTISSA_WIDTH.S - floor_exp_unbiased).asUInt
      val magnitude = (floor_sig >> shift_right).asSInt
      I := Mux(y_floor_bf16.sign, -magnitude, magnitude)
  }

  sub_unit_B.io.a := y
  sub_unit_B.io.b := y_floor_bf16
  sub_unit_B.io.sub := true.B
  val F = sub_unit_B.io.result

  val pow2_F = Wire(new BFloat16())
  val f_exp_unbiased = F.exponent.zext - BIAS.S
  when((F.exponent === 0.U && F.mantissa === 0.U) || F.sign) {
  //when((F.exponent === 0.U && F.mantissa === 0.U)) {
    pow2_F := BFloat16(ONE_BF16)
  } .elsewhen(f_exp_unbiased === -1.S) {
    val lut_index = (64.U + (F.mantissa >> 1.U))(6,0)
    pow2_F := BFloat16(lut_2_pow_F(lut_index))
  } .elsewhen(f_exp_unbiased < -1.S) {
    val shift_amount = (BIAS.U - F.exponent)
    val full_mantissa = Cat(1.U(1.W), F.mantissa)
    val lut_index = (full_mantissa >> shift_amount)(6,0)
    pow2_F := BFloat16(lut_2_pow_F(lut_index))
  } .otherwise {
    //pow2_F := BFloat16(ONE_BF16)
    pow2_F := BFloat16("h4000".U(16.W)) 
  }


  s4_q.io.enq.valid := s3_q.io.deq.valid
  s3_q.io.deq.ready := s4_q.io.enq.ready
  s4_q.io.enq.bits.I := I
  s4_q.io.enq.bits.pow2_F := pow2_F
  s4_q.io.enq.bits.is_special := s3_bits.is_special
  s4_q.io.enq.bits.special_result := s3_bits.special_result

  val s4_bits = s4_q.io.deq.bits
  val normal_result = Wire(new BFloat16())
  val pow2_F_exp_unbiased = s4_bits.pow2_F.exponent.zext - BIAS.S
  val final_exponent_unbiased = s4_bits.I + pow2_F_exp_unbiased
  val final_exponent_biased = final_exponent_unbiased + BIAS.S

  when(final_exponent_biased >= 255.S) {
    normal_result := inf(false.B)
  }.elsewhen(final_exponent_biased <= 0.S) {
    normal_result := zero(false.B)
  }.otherwise {
    normal_result.sign := false.B
    normal_result.exponent := final_exponent_biased.asUInt
    normal_result.mantissa := s4_bits.pow2_F.mantissa
  }

  val final_result = Mux(s4_bits.is_special, s4_bits.special_result, normal_result)

  s5_q.io.enq.valid := s4_q.io.deq.valid
  s4_q.io.deq.ready := s5_q.io.enq.ready
  s5_q.io.enq.bits := final_result
}


//in -> in_slice -> forward_mul_ln2flip -> res_mul_1_slice ->out_slice -> out
class BF16ExppUnit extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new BFloat16()))
    val out = Decoupled(new BFloat16())
  })

  import BF16ExpUnit._

  //============ dff1: capture the input ============
  val in_slice = Module(new ForwardSlice(16))
  in_slice.io.valid_src := io.in.valid
  in_slice.io.payload_src := io.in.bits.asBits()
  io.in.ready := in_slice.io.ready_src


  val in_bits_raw = Mux(in_slice.io.valid_dst, in_slice.io.payload_dst, 0.U(16.W))
  val in_bits = BFloat16(in_bits_raw)
  val sign = in_bits.sign
  val exponent = in_bits.exponent
  val mantissa = in_bits.mantissa

  val in_zero = WireDefault(false.B)
  val in_denormal = WireDefault(false.B)
  val in_inf = WireDefault(false.B)
  val in_nan = WireDefault(false.B)

  val too_small_L = 0xc386.U(16.W)
  val too_small_H = 0xff7f.U(16.W)
  val too_large_L = 0x42b2.U(16.W)
  val too_large_H = 0x7f7f.U(16.W)
  val bool_too_small = WireDefault(false.B)
  val bool_too_large = WireDefault(false.B)
  bool_too_small := (in_bits_raw >= too_small_L) & (in_bits_raw <= too_small_H)
  bool_too_large := (in_bits_raw >= too_large_L) & (in_bits_raw <= too_large_H)

  in_zero     := exponent === 0.U && mantissa === 0.U
  in_denormal := exponent === 0.U && mantissa =/= 0.U                         
  in_inf      := (exponent === 255.U && mantissa === 0.U) || bool_too_large || bool_too_small
  in_nan      := exponent === 255.U && mantissa =/= 0.U                         

  val Bias     = 127.U
  val ln2_flip = 23637.U(15.W)//(1/ln2) * 2^14 ≈ 23637/16384
  val Alpha    = 4.U(3.W) //0.25
  val Beta     = 7.U(4.W) //0.4375
  val Gamma1   = 363.U(9.W)//2.835937500 ≈ 363/128
  val Gamma2   = 278.U(9.W)//2.1679687500 ≈ 278/128

  val mant_with_1 = WireDefault(0.U(8.W))
  mant_with_1 := Cat(1.U(1.W), mantissa)
  val mant_mul_ln2flip = WireDefault(0.U(23.W))
  mant_mul_ln2flip := mant_with_1 * ln2_flip //8bit*15bit=23bit

  //============ dff2: dff after mul_ln2flip ============
  val forward_mul_ln2flip = Module(new ForwardSlice(35))
  val mant_mul_ln2flip_valid = WireDefault(false.B)
  val forward_mul_ln2flip_payload = WireDefault(0.U(35.W))// exponent | in_3 | sign | mant_mul_ln2flip
  val mant_mul_ln2flip_reg = WireDefault(0.U(23.W))
  val sign_reg = WireDefault(false.B)
  val in_3_reg = WireDefault(0.U(3.W))
  val exponent_reg = WireDefault(0.U(8.W))

  forward_mul_ln2flip.io.valid_src := in_slice.io.valid_dst
  forward_mul_ln2flip.io.payload_src := Cat(exponent, (in_zero || in_denormal).asUInt,in_inf.asUInt,in_nan.asUInt,sign.asUInt, mant_mul_ln2flip)
  in_slice.io.ready_dst := forward_mul_ln2flip.io.ready_src
  
  mant_mul_ln2flip_valid := forward_mul_ln2flip.io.valid_dst
  forward_mul_ln2flip_payload   := forward_mul_ln2flip.io.payload_dst
  

  mant_mul_ln2flip_reg := forward_mul_ln2flip_payload(22,0)
  sign_reg := forward_mul_ln2flip_payload(23)
  in_3_reg := forward_mul_ln2flip_payload(26,24)
  exponent_reg := forward_mul_ln2flip_payload(34,27)
  

  val shm = WireDefault(0.U(30.W))
  shm := Mux(exponent_reg >= Bias,
    mant_mul_ln2flip_reg << (exponent_reg - Bias),
    mant_mul_ln2flip_reg >> (Bias - exponent_reg)
  )

  val shm_nofraction = WireDefault(0.U(16.W))
  shm_nofraction := Mux(
    shm(13),
    ((shm >> 14) + 1.U),
    (shm >> 14) )
  val shm_nofraction_sign = WireDefault(0.U(16.W))
  shm_nofraction_sign := Mux(sign_reg, ~shm_nofraction + 1.U, shm_nofraction)

  val ne_temp = WireDefault(0.S(9.W))
  ne_temp := (shm_nofraction_sign.asSInt >> 7) + 127.S
  val overflow_up = WireDefault(false.B)
  val overflow_down = WireDefault(false.B)

  overflow_up := ne_temp >= 255.S
  overflow_down := ne_temp <= 0.S

  val nm = WireDefault(0.U(7.W))
  val ne = WireDefault(0.U(8.W))
  val frac_msb = WireDefault(false.B)
  nm := shm_nofraction_sign(6,0)
  ne := ne_temp(7,0)
  frac_msb := nm(6)
  
  val res_add_1 = WireDefault(0.U(9.W))
  val mant_mul = WireDefault(0.U(8.W))
  val res_mul_1 = WireDefault(0.U(11.W))
  res_add_1 := Mux(frac_msb, nm + Gamma2, nm + Gamma1)
  mant_mul := nm << 1
  res_mul_1 := Mux(frac_msb, (0xFF.U - mant_mul) * Beta,mant_mul * Alpha)

  //============ dff3: dff after res_mul_1 ============
  val res_mul_1_slice = Module(new ForwardSlice(35))
  val res_mul_1_valid = WireDefault(false.B)
  val res_mul_1_payload = WireDefault(0.U(35.W))
  val res_mul_1_reg = WireDefault(0.U(11.W))
  val res_add_1_reg = WireDefault(0.U(9.W))
  val overflow_up_reg = WireDefault(false.B)
  val overflow_down_reg = WireDefault(false.B)
  val frac_msb_reg = WireDefault(false.B)
  val ne_reg = WireDefault(0.U(8.W))
  val in_3_reg2 = WireDefault(0.U(3.W))
  val sign_reg2 = WireDefault(false.B)


  res_mul_1_slice.io.valid_src   := forward_mul_ln2flip.io.valid_dst
  res_mul_1_slice.io.payload_src := Cat(sign_reg,in_3_reg,ne,frac_msb,overflow_up,overflow_down,res_add_1,res_mul_1)//1+3+8+1+1+1+9+11=35bit
  forward_mul_ln2flip.io.ready_dst := res_mul_1_slice.io.ready_src

  res_mul_1_valid   := res_mul_1_slice.io.valid_dst
  res_mul_1_payload := res_mul_1_slice.io.payload_dst
  res_mul_1_reg := res_mul_1_payload(10,0)
  res_add_1_reg := res_mul_1_payload(19,11)
  overflow_down_reg := res_mul_1_payload(20)
  overflow_up_reg := res_mul_1_payload(21)
  frac_msb_reg := res_mul_1_payload(22)
  ne_reg := res_mul_1_payload(30,23)
  in_3_reg2 := res_mul_1_payload(33,31)
  sign_reg2 := res_mul_1_payload(34)
  
  val res_mul_2 = WireDefault(0.U(8.W))
  val result_mant = WireDefault(0.U(7.W))
  res_mul_2 := (res_add_1_reg * res_mul_1_reg) >> 12.U //9bit*11bit=20bit   20-12=8bit
  result_mant := Mux(frac_msb_reg,0x7F.U - res_mul_2, res_mul_2)//7bit
  val result_exp = ne_reg
  val result_sign = false.B
  val normal_result = Wire(new BFloat16())

  normal_result.sign := result_sign
  normal_result.exponent := result_exp
  normal_result.mantissa := result_mant

  val out_slice = Module(new ForwardSlice(16))
  out_slice.io.valid_src := res_mul_1_valid
  out_slice.io.payload_src := MuxCase(normal_result.asBits(),Seq(
    (in_3_reg2.orR)     -> MuxLookup(in_3_reg2, BFloat16(ONE_BF16))(Seq(
                                4.U -> BFloat16(ONE_BF16),
                                2.U -> Mux(sign_reg2,zero(false.B),inf(false.B)),
                                1.U -> nan()
                              )).asBits(),
    (overflow_up_reg)   -> Mux(sign_reg2,zero(false.B),inf(false.B)).asBits(),
    (overflow_down_reg) -> zero(false.B).asBits()
  ))
  res_mul_1_slice.io.ready_dst := out_slice.io.ready_src
  
  io.out.valid := out_slice.io.valid_dst
  io.out.bits  := BFloat16(out_slice.io.payload_dst)
  out_slice.io.ready_dst := io.out.ready

  in_slice.io.clk := clock
  in_slice.io.rst_n := !reset.asBool
  forward_mul_ln2flip.io.clk := clock
  forward_mul_ln2flip.io.rst_n := !reset.asBool
  res_mul_1_slice.io.clk := clock
  res_mul_1_slice.io.rst_n := !reset.asBool
  out_slice.io.clk := clock
  out_slice.io.rst_n := !reset.asBool
}


class VecExp_PacketIn(val VEC_LEN: Int) extends Bundle {
  val data = Vec(VEC_LEN, new BFloat16())
  //val ele_mask = Vec(VEC_LEN, Bool())
}

class VecExp_PacketOut(val VEC_LEN: Int) extends Bundle {
  val data = Vec(VEC_LEN, new BFloat16())
}

// 向量化Exp处理模块
class VectorExpUnit(val VEC_LEN: Int) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new VecExp_PacketIn(VEC_LEN)))
    val out = Decoupled(new VecExp_PacketOut(VEC_LEN))
  })
  
  val exp_units = Seq.fill(VEC_LEN)(Module(new BF16ExppUnit))
  // val out_mask = RegInit(VecInit(Seq.fill(VEC_LEN)(true.B)))
  // when(io.in.fire) {
  //   out_mask := io.in.bits.ele_mask
  // }
  
  // 输入分发到各个exp单元
  for (i <- 0 until VEC_LEN) {
    // exp_units(i).io.in.valid := io.in.valid && io.in.bits.ele_mask(i)
    // exp_units(i).io.in.bits := io.in.bits.data(i)
    // exp_units(i).io.out.ready := io.out.ready && out_mask(i)
    exp_units(i).io.in.valid := io.in.valid
    exp_units(i).io.in.bits := io.in.bits.data(i)
    exp_units(i).io.out.ready := io.out.ready
  }

  io.in.ready := VecInit((0 until VEC_LEN).map(i => 
      //exp_units(i).io.in.ready || !io.in.bits.ele_mask(i)
      exp_units(i).io.in.ready
  )).reduce(_ && _)

  io.out.valid := VecInit((0 until VEC_LEN).map(i => 
      //exp_units(i).io.out.valid || !out_mask(i)
      exp_units(i).io.out.valid
  )).reduce(_ && _)

  io.out.bits.data := VecInit(exp_units.map(_.io.out.bits))
} 



object ExpGenerator extends App {
  ChiselStage.emitSystemVerilogFile(
    new BF16ExpUnit,
    Array("--target-dir", "gen/module_test/exp"),
    Array("-disable-all-randomization", "-strip-debug-info")
  )
}

object ExppGenerator extends App {
  ChiselStage.emitSystemVerilogFile(
    new BF16ExppUnit,
    Array("--target-dir", "gen/module_test/expp"),
    Array("-disable-all-randomization", "-strip-debug-info")
  )
}

