import chisel3._

import Arithmetic.SIntArithmetic
import hardfloat._

// -----------------------
// Component Mixin Configs
// -----------------------

object GemminiConfigs {
  lazy val defaultConfig = GemminiArrayConfig[SInt, Float, Float](
    // Datatypes
    inputType = SInt(8.W),
    accType = SInt(32.W),

    spatialArrayOutputType = SInt(20.W),

    // Spatial array size options
    tileRows = 1,
    tileColumns = 1,
    meshRows = 16,
    meshColumns = 16,

    // Spatial array PE options
    dataflow = Dataflow.BOTH,

    // Scratchpad and accumulator
    sp_capacity = CapacityInKilobytes(256),
    acc_capacity = CapacityInKilobytes(64),

    sp_banks = 4,
    acc_banks = 2,

    sp_singleported = true,
    acc_singleported = false,

    // DNN options
    has_training_convs = true,
    has_max_pool = true,
    has_nonlinear_activations = true,

    // Reservation station entries
    reservation_station_entries_ld = 8,
    reservation_station_entries_st = 4,
    reservation_station_entries_ex = 16,

    // Ld/Ex/St instruction queue lengths
    ld_queue_length = 8,
    st_queue_length = 2,
    ex_queue_length = 8,

    // DMA options
    max_in_flight_mem_reqs = 16,

    dma_maxbytes = 64,
    dma_buswidth = 128,

    // TLB options
    tlb_size = 4,

    // Mvin and Accumulator scalar multiply options
    mvin_scale_args = Some(ScaleArguments(
      (t: SInt, f: Float) => {
        val f_rec = recFNFromFN(f.expWidth, f.sigWidth, f.bits)

        val in_to_rec_fn = Module(new INToRecFN(t.getWidth, f.expWidth, f.sigWidth))
        in_to_rec_fn.io.signedIn := true.B
        in_to_rec_fn.io.in := t.asTypeOf(UInt(t.getWidth.W))
        in_to_rec_fn.io.roundingMode := consts.round_near_even
        in_to_rec_fn.io.detectTininess := consts.tininess_afterRounding

        val t_rec = in_to_rec_fn.io.out

        val muladder = Module(new MulAddRecFN(f.expWidth, f.sigWidth))
        muladder.io.op := 0.U
        muladder.io.roundingMode := consts.round_near_even
        muladder.io.detectTininess := consts.tininess_afterRounding

        muladder.io.a := t_rec
        muladder.io.b := f_rec
        muladder.io.c := 0.U

        val rec_fn_to_in = Module(new RecFNToIN(f.expWidth, f.sigWidth, t.getWidth))
        rec_fn_to_in.io.in := muladder.io.out
        rec_fn_to_in.io.roundingMode := consts.round_near_even
        rec_fn_to_in.io.signedOut := true.B

        val overflow = rec_fn_to_in.io.intExceptionFlags(1)
        val maxsat = ((1 << (t.getWidth-1))-1).S
        val minsat = (-(1 << (t.getWidth-1))).S
        val sign = rawFloatFromRecFN(f.expWidth, f.sigWidth, rec_fn_to_in.io.in).sign
        val sat = Mux(sign, minsat, maxsat)

        Mux(overflow, sat, rec_fn_to_in.io.out.asTypeOf(t))
      },
      4, Float(8, 24), 4,
      identity = "1.0",
      c_str = "({float y = ROUND_NEAR_EVEN((x) * (scale)); y > INT8_MAX ? INT8_MAX : (y < INT8_MIN ? INT8_MIN : (elem_t)y);})"
    )),

    mvin_scale_acc_args = None,
    mvin_scale_shared = false,

    acc_scale_args = Some(ScaleArguments(
      (t: SInt, f: Float) => {
        val f_rec = recFNFromFN(f.expWidth, f.sigWidth, f.bits)

        val in_to_rec_fn = Module(new INToRecFN(t.getWidth, f.expWidth, f.sigWidth))
        in_to_rec_fn.io.signedIn := true.B
        in_to_rec_fn.io.in := t.asTypeOf(UInt(t.getWidth.W))
        in_to_rec_fn.io.roundingMode := consts.round_near_even
        in_to_rec_fn.io.detectTininess := consts.tininess_afterRounding

        val t_rec = in_to_rec_fn.io.out

        val muladder = Module(new MulAddRecFN(f.expWidth, f.sigWidth))
        muladder.io.op := 0.U
        muladder.io.roundingMode := consts.round_near_even
        muladder.io.detectTininess := consts.tininess_afterRounding

        muladder.io.a := t_rec
        muladder.io.b := f_rec
        muladder.io.c := 0.U

        val rec_fn_to_in = Module(new RecFNToIN(f.expWidth, f.sigWidth, t.getWidth))
        rec_fn_to_in.io.in := muladder.io.out
        rec_fn_to_in.io.roundingMode := consts.round_near_even
        rec_fn_to_in.io.signedOut := true.B

        val overflow = rec_fn_to_in.io.intExceptionFlags(1)
        val maxsat = ((1 << (t.getWidth-1))-1).S
        val minsat = (-(1 << (t.getWidth-1))).S
        val sign = rawFloatFromRecFN(f.expWidth, f.sigWidth, rec_fn_to_in.io.in).sign
        val sat = Mux(sign, minsat, maxsat)

        Mux(overflow, sat, rec_fn_to_in.io.out.asTypeOf(t))
      },
      8, Float(8, 24), -1,
      identity = "1.0",
      c_str = "({float y = ROUND_NEAR_EVEN((x) * (scale)); y > INT8_MAX ? INT8_MAX : (y < INT8_MIN ? INT8_MIN : (acc_t)y);})"
    )),

    // SoC counters options
    num_counter = 8,

    // Scratchpad and Accumulator input/output options
    acc_read_full_width = true,
    acc_read_small_width = true,

    ex_read_from_spad = true,
    ex_read_from_acc = true,
    ex_write_to_spad = true,
    ex_write_to_acc = true,
  )

  lazy val dummyConfig = GemminiArrayConfig[DummySInt, Float, Float](
    inputType = DummySInt(8),
    accType = DummySInt(32),
    spatialArrayOutputType = DummySInt(20),
    tileRows     = defaultConfig.tileRows,
    tileColumns  = defaultConfig.tileColumns,
    meshRows     = defaultConfig.meshRows,
    meshColumns  = defaultConfig.meshColumns,
    dataflow     = defaultConfig.dataflow,
    sp_capacity  = CapacityInKilobytes(128),
    acc_capacity = CapacityInKilobytes(128),
    sp_banks     = defaultConfig.sp_banks,
    acc_banks    = defaultConfig.acc_banks,
    sp_singleported = defaultConfig.sp_singleported,
    acc_singleported = defaultConfig.acc_singleported,
    has_training_convs = false,
    has_max_pool = defaultConfig.has_max_pool,
    has_nonlinear_activations = false,
    reservation_station_entries_ld = defaultConfig.reservation_station_entries_ld,
    reservation_station_entries_st = defaultConfig.reservation_station_entries_st,
    reservation_station_entries_ex = defaultConfig.reservation_station_entries_ex,
    ld_queue_length = defaultConfig.ld_queue_length,
    st_queue_length = defaultConfig.st_queue_length,
    ex_queue_length = defaultConfig.ex_queue_length,
    max_in_flight_mem_reqs = defaultConfig.max_in_flight_mem_reqs,
    dma_maxbytes = defaultConfig.dma_maxbytes,
    dma_buswidth = defaultConfig.dma_buswidth,
    tlb_size = defaultConfig.tlb_size,

    mvin_scale_args = Some(ScaleArguments(
      (t: DummySInt, f: Float) => t.dontCare,
      4, Float(8, 24), 4,
      identity = "1.0",
      c_str = "({float y = ROUND_NEAR_EVEN((x) * (scale)); y > INT8_MAX ? INT8_MAX : (y < INT8_MIN ? INT8_MIN : (elem_t)y);})"
    )),

    mvin_scale_acc_args = None,
    mvin_scale_shared = defaultConfig.mvin_scale_shared,

    acc_scale_args = Some(ScaleArguments(
      (t: DummySInt, f: Float) => t.dontCare,
      1, Float(8, 24), -1,
      identity = "1.0",
      c_str = "({float y = ROUND_NEAR_EVEN((x) * (scale)); y > INT8_MAX ? INT8_MAX : (y < INT8_MIN ? INT8_MIN : (acc_t)y);})"
    )),

    num_counter = 0,

    acc_read_full_width = false,
    acc_read_small_width = defaultConfig.acc_read_small_width,

    ex_read_from_spad = defaultConfig.ex_read_from_spad,
    ex_read_from_acc = false,
    ex_write_to_spad = false,
    ex_write_to_acc = defaultConfig.ex_write_to_acc,
  )

  lazy val chipConfig = defaultConfig.copy(sp_capacity=CapacityInKilobytes(64), acc_capacity=CapacityInKilobytes(32), dataflow=Dataflow.WS,
    acc_scale_args=Some(defaultConfig.acc_scale_args.get.copy(latency=4)),
    acc_singleported=true,
    acc_sub_banks=2,
    mesh_output_delay = 2,
    ex_read_from_acc=false,
    ex_write_to_spad=false,
    hardcode_d_to_garbage_addr = true
  )

  lazy val largeChipConfig = chipConfig.copy(sp_capacity=CapacityInKilobytes(128), acc_capacity=CapacityInKilobytes(64),
    tileRows=1, tileColumns=1,
    meshRows=32, meshColumns=32
  )

  lazy val leanConfig = defaultConfig.copy(dataflow=Dataflow.WS, max_in_flight_mem_reqs = 64, acc_read_full_width = false, ex_read_from_acc = false, ex_write_to_spad = false, hardcode_d_to_garbage_addr = true)

  lazy val leanPrintfConfig = defaultConfig.copy(dataflow=Dataflow.WS, max_in_flight_mem_reqs = 64, acc_read_full_width = false, ex_read_from_acc = false, ex_write_to_spad = false, hardcode_d_to_garbage_addr = true, use_firesim_simulation_counters=true)

}
