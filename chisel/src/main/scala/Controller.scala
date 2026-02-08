// FIXME: put gated clocks back in

import scala.math.{max, pow, sqrt}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import chisel3._
import chisel3.util._

import GemminiISA._
import Util._

class Gemmini[T <: Data: Arithmetic, U <: Data, V <: Data](config: GemminiArrayConfig[T, U, V]) extends Module {

  import config._

  val spad = Module(new Scratchpad(config))

  val io = IO(new Bundle {
    val cmd = Flipped(Decoupled(new RoCCCommand(xLen)))
    val resp = Decoupled(new RoCCResponse(xLen))
    val busy = Output(Bool())

    val readReqPacket = Decoupled(new ReadReqPacket(dma_maxbytes, max_in_flight_mem_reqs, coreMaxAddrBits))
    val readRespPacket = Flipped(Decoupled(new ReadRespPacket(dma_maxbytes, max_in_flight_mem_reqs, dma_buswidth)))

    val writeReqPacket = Decoupled(new WriteReqPacket(spad.data_width, coreMaxAddrBits))
    val writeRespPacket = Flipped(Decoupled(new WriteRespPacket(max_in_flight_mem_reqs, dma_maxbytes)))

    // val interrupt = Output(Bool())
    // val exception = Input(Bool())
  })

  io.readReqPacket <> spad.io.readReqPacket
  io.readRespPacket <> spad.io.readRespPacket
  io.writeReqPacket <> spad.io.writeReqPacket
  io.writeRespPacket <> spad.io.writeRespPacket


  val ext_mem_io = if (use_shared_ext_mem) Some(IO(new ExtSpadMemIO(sp_banks, acc_banks, acc_sub_banks))) else None
  ext_mem_io.foreach(_ <> spad.io.ext_mem.get)

  val tagWidth = 32

  // Counters
  val counters = Module(new CounterController(config.num_counter, config.xLen))
  io.resp <> counters.io.out  // Counter access command will be committed immediately
  counters.io.event_io.external_values(0) := 0.U
  counters.io.event_io.event_signal(0) := false.B
  counters.io.in.valid := false.B
  counters.io.in.bits := DontCare
  counters.io.event_io.collect(spad.io.counter)

  // TLB
  // implicit val edge = outer.spad.id_node.edges.out.head
  // val tlb = Module(new FrontendTLB(2, tlb_size, dma_maxbytes, use_tlb_register_filter, use_firesim_simulation_counters, use_shared_tlb))
  // (tlb.io.clients zip outer.spad.module.io.tlb).foreach(t => t._1 <> t._2)

  // tlb.io.exp.foreach(_.flush_skip := false.B)
  // tlb.io.exp.foreach(_.flush_retry := false.B)

  // io.ptw <> tlb.io.ptw

  // counters.io.event_io.collect(tlb.io.counter)
  counters.io.event_io.connectEventSignal(CounterEvent.DMA_TLB_HIT_REQ, false.B)
  counters.io.event_io.connectEventSignal(CounterEvent.DMA_TLB_TOTAL_REQ, false.B)
  counters.io.event_io.connectEventSignal(CounterEvent.DMA_TLB_MISS_CYCLE, false.B)

  spad.io.flush := DontCare

  val clock_en_reg = RegInit(true.B)
  // val gated_clock = if (clock_gate) ClockGate(clock, clock_en_reg, "gemmini_clock_gate") else clock
  // spad.module.clock := gated_clock

  /*
  //=========================================================================
  // Frontends: Incoming commands and ROB
  //=========================================================================

  // forward cmd to correct frontend. if the rob is busy, do not forward new
  // commands to tiler, and vice versa
  val is_cisc_mode = RegInit(false.B)

  val raw_cmd = Queue(io.cmd)
  val funct = raw_cmd.bits.inst.funct

  val is_cisc_funct = (funct === CISC_CONFIG) ||
                      (funct === ADDR_AB) ||
                      (funct === ADDR_CD) ||
                      (funct === SIZE_MN) ||
                      (funct === SIZE_K) ||
                      (funct === RPT_BIAS) ||
                      (funct === RESET) ||
                      (funct === COMPUTE_CISC)

  val raw_cisc_cmd = WireInit(raw_cmd)
  val raw_risc_cmd = WireInit(raw_cmd)
  raw_cisc_cmd.valid := false.B
  raw_risc_cmd.valid := false.B
  raw_cmd.ready := false.B

  //-------------------------------------------------------------------------
  // cisc
  val cmd_fsm = CmdFSM(outer.config)
  cmd_fsm.io.cmd <> raw_cisc_cmd
  val tiler = TilerController(outer.config)
  tiler.io.cmd_in <> cmd_fsm.io.tiler

  //-------------------------------------------------------------------------
  // risc
  val unrolled_cmd = LoopUnroller(raw_risc_cmd, outer.config.meshRows * outer.config.tileRows)
  */

  // val reservation_station = withClock (gated_clock) { Module(new ReservationStation(outer.config, new GemminiCmd(reservation_station_entries))) }
  val reservation_station = Module(new ReservationStation(config, new GemminiCmd(xLen, reservation_station_entries)))
  counters.io.event_io.collect(reservation_station.io.counter)

  when (io.cmd.valid && io.cmd.bits.inst.funct === CLKGATE_EN && !io.busy) {
    clock_en_reg := io.cmd.bits.rs1(0)
  }

  val raw_cmd_q = Module(new Queue(new GemminiCmd(xLen, reservation_station_entries), entries = 2))
  raw_cmd_q.io.enq.valid := io.cmd.valid
  io.cmd.ready := raw_cmd_q.io.enq.ready
  raw_cmd_q.io.enq.bits.cmd := io.cmd.bits
  raw_cmd_q.io.enq.bits.rob_id := DontCare
  raw_cmd_q.io.enq.bits.from_conv_fsm := false.B
  raw_cmd_q.io.enq.bits.from_matmul_fsm := false.B

  val raw_cmd = raw_cmd_q.io.deq

  val max_lds = reservation_station_entries_ld
  val max_exs = reservation_station_entries_ex
  val max_sts = reservation_station_entries_st

  val (conv_cmd, loop_conv_unroller_busy) = if (has_loop_conv) LoopConv(
      raw_cmd, reservation_station.io.conv_ld_completed, reservation_station.io.conv_st_completed, reservation_station.io.conv_ex_completed,
      meshRows*tileRows, coreMaxAddrBits, reservation_station_entries, max_lds, max_exs, max_sts, sp_banks * sp_bank_entries, acc_banks * acc_bank_entries,
      inputType.getWidth, accType.getWidth, dma_maxbytes, new ConfigMvinRs1(mvin_scale_t_bits, block_stride_bits, pixel_repeats_bits), new MvinRs2(mvin_rows_bits, mvin_cols_bits, local_addr_t),
      new ConfigMvoutRs2(acc_scale_t_bits, 32), new MvoutRs2(mvout_rows_bits, mvout_cols_bits, local_addr_t), new ConfigExRs1(acc_scale_t_bits),
      new PreloadRs(mvin_rows_bits, mvin_cols_bits, local_addr_t), new PreloadRs(mvout_rows_bits, mvout_cols_bits, local_addr_t), new ComputeRs(mvin_rows_bits, mvin_cols_bits, local_addr_t),
      new ComputeRs(mvin_rows_bits, mvin_cols_bits, local_addr_t), has_training_convs, has_max_pool, has_first_layer_optimizations, has_dw_convs, xLen)
  else (raw_cmd, false.B)

  val (loop_cmd, loop_matmul_unroller_busy) = LoopMatmul(if (has_loop_conv) conv_cmd else raw_cmd, reservation_station.io.matmul_ld_completed, reservation_station.io.matmul_st_completed, reservation_station.io.matmul_ex_completed,
    meshRows*tileRows, coreMaxAddrBits, reservation_station_entries, max_lds, max_exs, max_sts, sp_banks * sp_bank_entries, acc_banks * acc_bank_entries,
    inputType.getWidth, accType.getWidth, dma_maxbytes, new MvinRs2(mvin_rows_bits, mvin_cols_bits, local_addr_t),
    new PreloadRs(mvin_rows_bits, mvin_cols_bits, local_addr_t), new PreloadRs(mvout_rows_bits, mvout_cols_bits, local_addr_t),
    new ComputeRs(mvin_rows_bits, mvin_cols_bits, local_addr_t), new ComputeRs(mvin_rows_bits, mvin_cols_bits, local_addr_t),
    new MvoutRs2(mvout_rows_bits, mvout_cols_bits, local_addr_t), xLen)

  val unrolled_cmd = Queue(loop_cmd)
  unrolled_cmd.ready := false.B
  counters.io.event_io.connectEventSignal(CounterEvent.LOOP_MATMUL_ACTIVE_CYCLES, loop_matmul_unroller_busy)

  // Wire up controllers to ROB
  reservation_station.io.alloc.valid := false.B
  reservation_station.io.alloc.bits := unrolled_cmd.bits

  /*
  //-------------------------------------------------------------------------
  // finish muxing control signals to rob (risc) or tiler (cisc)
  when (raw_cmd.valid && is_cisc_funct && !rob.io.busy) {
    is_cisc_mode       := true.B
    raw_cisc_cmd.valid := true.B
    raw_cmd.ready      := raw_cisc_cmd.ready
  }
  .elsewhen (raw_cmd.valid && !is_cisc_funct && !tiler.io.busy) {
    is_cisc_mode       := false.B
    raw_risc_cmd.valid := true.B
    raw_cmd.ready      := raw_risc_cmd.ready
  }
  */

  //=========================================================================
  // Controllers
  //=========================================================================
  val load_controller = Module(new LoadController(config, coreMaxAddrBits, local_addr_t))
  val store_controller = Module(new StoreController(config, coreMaxAddrBits, local_addr_t))
  val ex_controller = Module(new ExecuteController(xLen, tagWidth, config))

  counters.io.event_io.collect(load_controller.io.counter)
  counters.io.event_io.collect(store_controller.io.counter)
  counters.io.event_io.collect(ex_controller.io.counter)

  /*
  tiler.io.issue.load.ready := false.B
  tiler.io.issue.store.ready := false.B
  tiler.io.issue.exec.ready := false.B
  */

  reservation_station.io.issue.ld.ready := false.B
  reservation_station.io.issue.st.ready := false.B
  reservation_station.io.issue.ex.ready := false.B

  /*
  when (is_cisc_mode) {
    load_controller.io.cmd  <> tiler.io.issue.load
    store_controller.io.cmd <> tiler.io.issue.store
    ex_controller.io.cmd  <> tiler.io.issue.exec
  }
  .otherwise {
    load_controller.io.cmd.valid := rob.io.issue.ld.valid
    rob.io.issue.ld.ready := load_controller.io.cmd.ready
    load_controller.io.cmd.bits.cmd := rob.io.issue.ld.cmd
    load_controller.io.cmd.bits.cmd.inst.funct := rob.io.issue.ld.cmd.inst.funct
    load_controller.io.cmd.bits.rob_id.push(rob.io.issue.ld.rob_id)

    store_controller.io.cmd.valid := rob.io.issue.st.valid
    rob.io.issue.st.ready := store_controller.io.cmd.ready
    store_controller.io.cmd.bits.cmd := rob.io.issue.st.cmd
    store_controller.io.cmd.bits.cmd.inst.funct := rob.io.issue.st.cmd.inst.funct
    store_controller.io.cmd.bits.rob_id.push(rob.io.issue.st.rob_id)

    ex_controller.io.cmd.valid := rob.io.issue.ex.valid
    rob.io.issue.ex.ready := ex_controller.io.cmd.ready
    ex_controller.io.cmd.bits.cmd := rob.io.issue.ex.cmd
    ex_controller.io.cmd.bits.cmd.inst.funct := rob.io.issue.ex.cmd.inst.funct
    ex_controller.io.cmd.bits.rob_id.push(rob.io.issue.ex.rob_id)
  }
  */

  load_controller.io.cmd.valid := reservation_station.io.issue.ld.valid
  reservation_station.io.issue.ld.ready := load_controller.io.cmd.ready
  load_controller.io.cmd.bits := reservation_station.io.issue.ld.cmd
  load_controller.io.cmd.bits.rob_id.push(reservation_station.io.issue.ld.rob_id)

  store_controller.io.cmd.valid := reservation_station.io.issue.st.valid
  reservation_station.io.issue.st.ready := store_controller.io.cmd.ready
  store_controller.io.cmd.bits := reservation_station.io.issue.st.cmd
  store_controller.io.cmd.bits.rob_id.push(reservation_station.io.issue.st.rob_id)

  ex_controller.io.cmd.valid := reservation_station.io.issue.ex.valid
  reservation_station.io.issue.ex.ready := ex_controller.io.cmd.ready
  ex_controller.io.cmd.bits := reservation_station.io.issue.ex.cmd
  ex_controller.io.cmd.bits.rob_id.push(reservation_station.io.issue.ex.rob_id)

  // Wire up scratchpad to controllers
  spad.io.dma.read <> load_controller.io.dma
  spad.io.dma.write <> store_controller.io.dma
  ex_controller.io.srams.read <> spad.io.srams.read
  ex_controller.io.srams.write <> spad.io.srams.write
  spad.io.acc.read_req <> ex_controller.io.acc.read_req
  ex_controller.io.acc.read_resp <> spad.io.acc.read_resp
  ex_controller.io.acc.write <> spad.io.acc.write

  // Im2Col unit
  val im2col = Module(new Im2Col(config))

  // Wire up Im2col
  counters.io.event_io.collect(im2col.io.counter)
  // im2col.io.sram_reads <> spad.module.io.srams.read
  im2col.io.req <> ex_controller.io.im2col.req
  ex_controller.io.im2col.resp <> im2col.io.resp

  // Wire arbiter for ExecuteController and Im2Col scratchpad reads
  (ex_controller.io.srams.read, im2col.io.sram_reads, spad.io.srams.read).zipped.foreach { case (ex_read, im2col_read, spad_read) =>
    val req_arb = Module(new Arbiter(new ScratchpadReadReq(n=sp_bank_entries), 2))

    req_arb.io.in(0) <> ex_read.req
    req_arb.io.in(1) <> im2col_read.req

    spad_read.req <> req_arb.io.out

    // TODO if necessary, change how the responses are handled when fromIm2Col is added to spad read interface

    ex_read.resp.valid := spad_read.resp.valid
    im2col_read.resp.valid := spad_read.resp.valid

    ex_read.resp.bits := spad_read.resp.bits
    im2col_read.resp.bits := spad_read.resp.bits

    spad_read.resp.ready := ex_read.resp.ready || im2col_read.resp.ready
  }

  // Wire up controllers to ROB
  reservation_station.io.alloc.valid := false.B
  // rob.io.alloc.bits := compressed_cmd.bits
  reservation_station.io.alloc.bits := unrolled_cmd.bits

  /*
  //=========================================================================
  // committed insn return path to frontends
  //=========================================================================

  //-------------------------------------------------------------------------
  // cisc
  tiler.io.completed.exec.valid := ex_controller.io.completed.valid
  tiler.io.completed.exec.bits := ex_controller.io.completed.bits

  tiler.io.completed.load <> load_controller.io.completed
  tiler.io.completed.store <> store_controller.io.completed

  // mux with cisc frontend arbiter
  tiler.io.completed.exec.valid  := ex_controller.io.completed.valid && is_cisc_mode
  tiler.io.completed.load.valid  := load_controller.io.completed.valid && is_cisc_mode
  tiler.io.completed.store.valid := store_controller.io.completed.valid && is_cisc_mode
  */

  //-------------------------------------------------------------------------
  // risc
  val reservation_station_completed_arb = Module(new Arbiter(UInt(log2Up(reservation_station_entries).W), 3))

  reservation_station_completed_arb.io.in(0).valid := ex_controller.io.completed.valid
  reservation_station_completed_arb.io.in(0).bits := ex_controller.io.completed.bits

  reservation_station_completed_arb.io.in(1) <> load_controller.io.completed
  reservation_station_completed_arb.io.in(2) <> store_controller.io.completed

  // mux with cisc frontend arbiter
  reservation_station_completed_arb.io.in(0).valid := ex_controller.io.completed.valid // && !is_cisc_mode
  reservation_station_completed_arb.io.in(1).valid := load_controller.io.completed.valid // && !is_cisc_mode
  reservation_station_completed_arb.io.in(2).valid := store_controller.io.completed.valid // && !is_cisc_mode

  reservation_station.io.completed.valid := reservation_station_completed_arb.io.out.valid
  reservation_station.io.completed.bits := reservation_station_completed_arb.io.out.bits
  reservation_station_completed_arb.io.out.ready := true.B

  // Wire up global RoCC signals
  io.busy := raw_cmd.valid || loop_conv_unroller_busy || loop_matmul_unroller_busy || reservation_station.io.busy || spad.io.busy || unrolled_cmd.valid || loop_cmd.valid || conv_cmd.valid

  // io.interrupt := tlb.io.exp.map(_.interrupt).reduce(_ || _)

  // assert(!io.interrupt, "Interrupt handlers have not been written yet")

  // Cycle counters
  val incr_ld_cycles = load_controller.io.busy && !store_controller.io.busy && !ex_controller.io.busy
  val incr_st_cycles = !load_controller.io.busy && store_controller.io.busy && !ex_controller.io.busy
  val incr_ex_cycles = !load_controller.io.busy && !store_controller.io.busy && ex_controller.io.busy

  val incr_ld_st_cycles = load_controller.io.busy && store_controller.io.busy && !ex_controller.io.busy
  val incr_ld_ex_cycles = load_controller.io.busy && !store_controller.io.busy && ex_controller.io.busy
  val incr_st_ex_cycles = !load_controller.io.busy && store_controller.io.busy && ex_controller.io.busy

  val incr_ld_st_ex_cycles = load_controller.io.busy && store_controller.io.busy && ex_controller.io.busy

  counters.io.event_io.connectEventSignal(CounterEvent.MAIN_LD_CYCLES, incr_ld_cycles)
  counters.io.event_io.connectEventSignal(CounterEvent.MAIN_ST_CYCLES, incr_st_cycles)
  counters.io.event_io.connectEventSignal(CounterEvent.MAIN_EX_CYCLES, incr_ex_cycles)
  counters.io.event_io.connectEventSignal(CounterEvent.MAIN_LD_ST_CYCLES, incr_ld_st_cycles)
  counters.io.event_io.connectEventSignal(CounterEvent.MAIN_LD_EX_CYCLES, incr_ld_ex_cycles)
  counters.io.event_io.connectEventSignal(CounterEvent.MAIN_ST_EX_CYCLES, incr_st_ex_cycles)
  counters.io.event_io.connectEventSignal(CounterEvent.MAIN_LD_ST_EX_CYCLES, incr_ld_st_ex_cycles)

  // Issue commands to controllers
  // TODO we combinationally couple cmd.ready and cmd.valid signals here
  // when (compressed_cmd.valid) {
  when (unrolled_cmd.valid) {
    // val config_cmd_type = cmd.bits.rs1(1,0) // TODO magic numbers

    //val funct = unrolled_cmd.bits.inst.funct
    val risc_funct = unrolled_cmd.bits.cmd.inst.funct

    val is_flush = risc_funct === FLUSH_CMD
    val is_counter_op = risc_funct === COUNTER_OP
    val is_clock_gate_en = risc_funct === CLKGATE_EN

    /*
    val is_load = (funct === LOAD_CMD) || (funct === CONFIG_CMD && config_cmd_type === CONFIG_LOAD)
    val is_store = (funct === STORE_CMD) || (funct === CONFIG_CMD && config_cmd_type === CONFIG_STORE)
    val is_ex = (funct === COMPUTE_AND_FLIP_CMD || funct === COMPUTE_AND_STAY_CMD || funct === PRELOAD_CMD) ||
    (funct === CONFIG_CMD && config_cmd_type === CONFIG_EX)
    */

    when (is_flush) {
      // val skip = unrolled_cmd.bits.cmd.rs1(0)
      // tlb.io.exp.foreach(_.flush_skip := skip)
      // tlb.io.exp.foreach(_.flush_retry := !skip)

      unrolled_cmd.ready := true.B // TODO should we wait for an acknowledgement from the TLB?
    }

    .elsewhen (is_counter_op) {
      // If this is a counter access/configuration command, execute immediately
      counters.io.in.valid := unrolled_cmd.valid
      unrolled_cmd.ready := counters.io.in.ready
      counters.io.in.bits := unrolled_cmd.bits.cmd
    }

    .elsewhen (is_clock_gate_en) {
      unrolled_cmd.ready := true.B
    }

    .otherwise {
      reservation_station.io.alloc.valid := true.B

      when(reservation_station.io.alloc.fire) {
        // compressed_cmd.ready := true.B
        unrolled_cmd.ready := true.B
      }
    }
  }

  // Debugging signals
  val pipeline_stall_counter = RegInit(0.U(32.W))
  when (io.cmd.fire) {
    pipeline_stall_counter := 0.U
  }.elsewhen(io.busy) {
    pipeline_stall_counter := pipeline_stall_counter + 1.U
  }
  assert(pipeline_stall_counter < 10000000.U, "pipeline stall")

  /*
  //=========================================================================
  // Wire up global RoCC signals
  //=========================================================================
  io.busy := raw_cmd.valid || unrolled_cmd.valid || rob.io.busy || spad.module.io.busy || tiler.io.busy
  io.interrupt := tlb.io.exp.interrupt

  // hack
  when(is_cisc_mode || !(unrolled_cmd.valid || rob.io.busy || tiler.io.busy)){
    tlb.io.exp.flush_retry := cmd_fsm.io.flush_retry
    tlb.io.exp.flush_skip  := cmd_fsm.io.flush_skip
  }
  */

  //=========================================================================
  // Performance Counters Access
  //=========================================================================

}

object GemminiMain extends App {
  println("Generating Gemmini hardware")

  val input_type = Float(8, 24)
  val output_type = Float(8, 24)
  val acc_type = Float(8, 24)
  // val input_type = SInt(32.W)
  // val output_type = SInt(32.W)
  // val acc_type = SInt(32.W)

  // val dataflow = Dataflow.WS
  // val meshColumns = 8
  // val tileColumns = 1
  // val meshRows = 8
  // val tileRows = 1
  // val block_size = meshRows * tileRows
  // val output_delay = 0
  // val tile_latency = 0
  // val tree_reduction = false
  // val left_banks = 1
  // val up_banks = 1
  // val out_banks = 1

  // val sp_banks = 4
  // val ms = 2
  // val sp_bank_entries = ms * meshRows * tileRows / sp_banks
  // val acc_banks = 2
  // val acc_bank_entries = ms * meshRows * tileRows / acc_banks

  // val local_addr_t = new LocalAddr(sp_banks, sp_bank_entries, acc_banks, acc_bank_entries)

  // val reservation_station_entries_ld = 8
  // val reservation_station_entries_st = 4
  // val reservation_station_entries_ex = 16
  // val res_max_per_type = max(reservation_station_entries_ld,
  //   max(reservation_station_entries_st, reservation_station_entries_ex))
  // val reservation_station_entries = res_max_per_type * 3

  // val mesh_tag = new Bundle with TagQueueTag {
  //   val rob_id = UDValid(UInt(log2Up(reservation_station_entries).W))
  //   val addr = local_addr_t.cloneType
  //   val rows = UInt(log2Up(block_size + 1).W)
  //   val cols = UInt(log2Up(block_size + 1).W)

  //   override def make_this_garbage(dummy: Int = 0): Unit = {
  //     rob_id.valid := false.B
  //     addr.make_this_garbage()
  //   }
  // }

  emitVerilog(new Gemmini(GemminiConfigs.defaultConfig), args)

  // emitVerilog(new MeshWithDelays(input_type, output_type, acc_type, /* tagType */ mesh_tag,
  //   dataflow, tree_reduction, tile_latency, output_delay, tileRows, tileColumns, meshRows, meshColumns,
  //   left_banks, up_banks, out_banks, /* n_simultaneous_matmuls */ -1), args)
}
