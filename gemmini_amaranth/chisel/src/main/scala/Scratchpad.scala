import chisel3._
import chisel3.util._

import Util._

class ScratchpadMemReadRequest[U <: Data](local_addr_t: LocalAddr, scale_t_bits: Int, coreMaxAddrBits: Int) extends Bundle {
  val vaddr = UInt(coreMaxAddrBits.W)
  val laddr = local_addr_t.cloneType

  val cols = UInt(16.W) // TODO don't use a magic number for the width here
  val repeats = UInt(16.W) // TODO don't use a magic number for the width here
  val scale = UInt(scale_t_bits.W)
  val has_acc_bitwidth = Bool()
  val all_zeros = Bool()
  val block_stride = UInt(16.W) // TODO magic numbers
  val pixel_repeats = UInt(8.W) // TODO magic numbers
  val cmd_id = UInt(8.W) // TODO don't use a magic number here
  val status = new MStatus
}

class ScratchpadMemWriteRequest(local_addr_t: LocalAddr, acc_t_bits: Int, scale_t_bits: Int, coreMaxAddrBits: Int) extends Bundle {
  val vaddr = UInt(coreMaxAddrBits.W)
  val laddr = local_addr_t.cloneType

  val acc_act = UInt(Activation.bitwidth.W) // TODO don't use a magic number for the width here
  val acc_scale = UInt(scale_t_bits.W)
  val acc_igelu_qb = UInt(acc_t_bits.W)
  val acc_igelu_qc = UInt(acc_t_bits.W)
  val acc_iexp_qln2 = UInt(acc_t_bits.W)
  val acc_iexp_qln2_inv = UInt(acc_t_bits.W)
  val acc_norm_stats_id = UInt(8.W) // TODO magic number

  val len = UInt(16.W) // TODO don't use a magic number for the width here
  val block = UInt(8.W) // TODO don't use a magic number for the width here

  val cmd_id = UInt(8.W) // TODO don't use a magic number here
  val status = new MStatus

  // Pooling variables
  val pool_en = Bool()
  val store_en = Bool()

}

class ScratchpadMemWriteResponse extends Bundle {
  val cmd_id = UInt(8.W) // TODO don't use a magic number here
}

class ScratchpadMemReadResponse extends Bundle {
  val bytesRead = UInt(16.W) // TODO magic number here
  val cmd_id = UInt(8.W) // TODO don't use a magic number here
}

class ScratchpadReadMemIO[U <: Data](local_addr_t: LocalAddr, scale_t_bits: Int, coreMaxAddrBits: Int) extends Bundle {
  val req = Decoupled(new ScratchpadMemReadRequest(local_addr_t, scale_t_bits, coreMaxAddrBits))
  val resp = Flipped(Valid(new ScratchpadMemReadResponse))
}

class ScratchpadWriteMemIO(local_addr_t: LocalAddr, acc_t_bits: Int, scale_t_bits: Int, coreMaxAddrBits: Int) extends Bundle {
  val req = Decoupled(new ScratchpadMemWriteRequest(local_addr_t, acc_t_bits, scale_t_bits, coreMaxAddrBits))
  val resp = Flipped(Valid(new ScratchpadMemWriteResponse))
}

class ScratchpadReadReq(val n: Int) extends Bundle {
  val addr = UInt(log2Ceil(n).W)
  val fromDMA = Bool()
}

class ScratchpadReadResp(val w: Int) extends Bundle {
  val data = UInt(w.W)
  val fromDMA = Bool()
}

class ScratchpadReadIO(val n: Int, val w: Int) extends Bundle {
  val req = Decoupled(new ScratchpadReadReq(n))
  val resp = Flipped(Decoupled(new ScratchpadReadResp(w)))
}

class ScratchpadWriteIO(val n: Int, val w: Int, val mask_len: Int) extends Bundle {
  val en = Output(Bool())
  val addr = Output(UInt(log2Ceil(n).W))
  val mask = Output(Vec(mask_len, Bool()))
  val data = Output(UInt(w.W))
}

class ReadReqPacket(maxBytes: Int, nXacts: Int, vaddrBits: Int) extends Bundle {
  val lg_size = UInt(log2Ceil(log2Ceil(maxBytes+1)).W)
  val bytes_read = UInt(log2Up(maxBytes+1).W)
  val shift = UInt(log2Up(maxBytes).W)
  val vaddr = UInt(vaddrBits.W)
  val source = UInt(log2Up(nXacts).W)
}

class ReadRespPacket(maxBytes: Int, nXacts: Int, beatBits: Int) extends Bundle {
  val source = UInt(log2Up(nXacts).W)
  val data = UInt(beatBits.W)
  val lg_size = UInt(log2Up(log2Up(maxBytes+1)+1).W)
  val last = Bool()
}

class WriteReqPacket(dataWidth: Int, vaddrBits: Int) extends Bundle {
  val data = UInt(dataWidth.W)
  val vaddr = UInt(vaddrBits.W)
  val last = Bool()
}

class WriteRespPacket(nXacts: Int, maxBytes: Int) extends Bundle {
  val lg_size = UInt(log2Up(log2Up(maxBytes+1)+1).W)
  val source = UInt(log2Up(nXacts).W)
}

class ScratchpadBank(n: Int, w: Int, aligned_to: Int, single_ported: Boolean, use_shared_ext_mem: Boolean, is_dummy: Boolean, coreMaxAddrBits: Int) extends Module {
  // This is essentially a pipelined SRAM with the ability to stall pipeline stages

  require(w % aligned_to == 0 || w < aligned_to)
  val mask_len = (w / (aligned_to * 8)) max 1 // How many mask bits are there?
  val mask_elem = UInt((w min (aligned_to * 8)).W) // What datatype does each mask bit correspond to?

  val io = IO(new Bundle {
    val read = Flipped(new ScratchpadReadIO(n, w))
    val write = Flipped(new ScratchpadWriteIO(n, w, mask_len))
    val ext_mem = if (use_shared_ext_mem) Some(new ExtMemIO) else None
  })

  val (read, write) = if (is_dummy) {
    def read(addr: UInt, ren: Bool): Data = 0.U
    def write(addr: UInt, wdata: Vec[UInt], wmask: Vec[Bool]): Unit = { }
    (read _, write _)
  } else if (use_shared_ext_mem) {
    def read(addr: UInt, ren: Bool): Data = {
      io.ext_mem.get.read_en := ren
      io.ext_mem.get.read_addr := addr
      io.ext_mem.get.read_data
    }
    io.ext_mem.get.write_en := false.B
    io.ext_mem.get.write_addr := DontCare
    io.ext_mem.get.write_data := DontCare
    io.ext_mem.get.write_mask := DontCare
    def write(addr: UInt, wdata: Vec[UInt], wmask: Vec[Bool]) = {
      io.ext_mem.get.write_en := true.B
      io.ext_mem.get.write_addr := addr
      io.ext_mem.get.write_data := wdata.asUInt
      io.ext_mem.get.write_mask := wmask.asUInt
    }
    (read _, write _)
  } else {
    val mem = SyncReadMem(n, Vec(mask_len, mask_elem))
    def read(addr: UInt, ren: Bool): Data = mem.read(addr, ren)
    def write(addr: UInt, wdata: Vec[UInt], wmask: Vec[Bool]) = mem.write(addr, wdata, wmask)
    (read _, write _)
  }

  // When the scratchpad is single-ported, the writes take precedence
  val singleport_busy_with_write = single_ported.B && io.write.en

  when (io.write.en) {
    if (aligned_to >= w)
      write(io.write.addr, io.write.data.asTypeOf(Vec(mask_len, mask_elem)), VecInit((~(0.U(mask_len.W))).asBools))
    else
      write(io.write.addr, io.write.data.asTypeOf(Vec(mask_len, mask_elem)), io.write.mask)
  }

  val raddr = io.read.req.bits.addr
  val ren = io.read.req.fire
  val rdata = if (single_ported) {
    assert(!(ren && io.write.en))
    read(raddr, ren && !io.write.en).asUInt
  } else {
    read(raddr, ren).asUInt
  }

  val fromDMA = io.read.req.bits.fromDMA

  // Make a queue which buffers the result of an SRAM read if it can't immediately be consumed
  val q = Module(new Queue(new ScratchpadReadResp(w), 1, true, true))
  q.io.enq.valid := RegNext(ren)
  q.io.enq.bits.data := rdata
  q.io.enq.bits.fromDMA := RegNext(fromDMA)

  val q_will_be_empty = (q.io.count +& q.io.enq.fire) - q.io.deq.fire === 0.U
  io.read.req.ready := q_will_be_empty && !singleport_busy_with_write

  io.read.resp <> q.io.deq
}


class Scratchpad[T <: Data, U <: Data, V <: Data](config: GemminiArrayConfig[T, U, V]) (implicit ev: Arithmetic[T]) extends Module {

  import config._
  import ev._

  val maxBytes = dma_maxbytes
  val dataBits = dma_buswidth

  val block_rows = meshRows * tileRows
  val block_cols = meshColumns * tileColumns
  val spad_w = inputType.getWidth *  block_cols
  val acc_w = accType.getWidth * block_cols

  val reader = Module(new StreamReader(config, max_in_flight_mem_reqs, dataBits, maxBytes, spad_w, acc_w, aligned_to,
    sp_banks * sp_bank_entries, acc_banks * acc_bank_entries, block_rows, use_tlb_register_filter,
    coreMaxAddrBits, coreMaxAddrBits))

  val data_width = if (acc_read_full_width) acc_w else spad_w
  val writer = Module(new StreamWriter(max_in_flight_mem_reqs, dataBits, maxBytes,
    data_width, aligned_to, inputType, block_cols, use_tlb_register_filter,
    coreMaxAddrBits, coreMaxAddrBits))

  val io = IO(new Bundle {
    // DMA ports
    val dma = new Bundle {
      val read = Flipped(new ScratchpadReadMemIO(local_addr_t, mvin_scale_t_bits, coreMaxAddrBits))
      val write = Flipped(new ScratchpadWriteMemIO(local_addr_t, accType.getWidth, acc_scale_t_bits, coreMaxAddrBits))
    }

    // SRAM ports
    val srams = new Bundle {
      val read = Flipped(Vec(sp_banks, new ScratchpadReadIO(sp_bank_entries, spad_w)))
      val write = Flipped(Vec(sp_banks, new ScratchpadWriteIO(sp_bank_entries, spad_w, (spad_w / (aligned_to * 8)) max 1)))
    }

    // Accumulator ports
    val acc = new Bundle {
      val read_req = Flipped(Vec(acc_banks, Decoupled(new AccumulatorReadReq(
        acc_bank_entries, accType, acc_scale_t.asInstanceOf[V]
      ))))
      val read_resp = Vec(acc_banks, Decoupled(new AccumulatorScaleResp(
        Vec(meshColumns, Vec(tileColumns, inputType)),
        Vec(meshColumns, Vec(tileColumns, accType))
      )))
      val write = Flipped(Vec(acc_banks, Decoupled(new AccumulatorWriteReq(
        acc_bank_entries, Vec(meshColumns, Vec(tileColumns, accType))
      ))))
    }

    val ext_mem = if (use_shared_ext_mem) {
      Some(new ExtSpadMemIO(sp_banks, acc_banks, acc_sub_banks))
    } else {
      None
    }

    val readReqPacket = Decoupled(new ReadReqPacket(maxBytes, max_in_flight_mem_reqs, coreMaxAddrBits))
    val readRespPacket = Flipped(Decoupled(new ReadRespPacket(maxBytes, max_in_flight_mem_reqs, dataBits)))

    val writeReqPacket = Decoupled(new WriteReqPacket(data_width, coreMaxAddrBits))
    val writeRespPacket = Flipped(Decoupled(new WriteRespPacket(max_in_flight_mem_reqs, maxBytes)))

    // Misc. ports
    val busy = Output(Bool())
    val flush = Input(Bool())
    val counter = new CounterEventIO()
  })

  io.readReqPacket <> reader.io.reqPacket
  io.readRespPacket <> reader.io.respPacket

  io.writeReqPacket <> writer.io.reqPacket
  io.writeRespPacket <> writer.io.respPacket

  val write_dispatch_q = Queue(io.dma.write.req)

  // Write norm/scale queues are necessary to maintain in-order requests to accumulator norm/scale units
  // Writes from main SPAD just flow directly between scale_q and issue_q, while writes
  // From acc are ordered
  val write_norm_q  = Module(new Queue(new ScratchpadMemWriteRequest(local_addr_t, accType.getWidth, acc_scale_t_bits, coreMaxAddrBits), spad_read_delay+2))
  val write_scale_q = Module(new Queue(new ScratchpadMemWriteRequest(local_addr_t, accType.getWidth, acc_scale_t_bits, coreMaxAddrBits), spad_read_delay+2))
  val write_issue_q = Module(new Queue(new ScratchpadMemWriteRequest(local_addr_t, accType.getWidth, acc_scale_t_bits, coreMaxAddrBits), spad_read_delay+1, pipe=true))
  val read_issue_q  = Module(new Queue(new ScratchpadMemReadRequest(local_addr_t, mvin_scale_t_bits, coreMaxAddrBits), spad_read_delay+1, pipe=true)) // TODO can't this just be a normal queue?


  write_dispatch_q.ready := false.B

  write_norm_q.io.enq.valid := false.B
  write_norm_q.io.enq.bits := write_dispatch_q.bits
  write_norm_q.io.deq.ready := false.B

  write_scale_q.io.enq.valid := false.B
  write_scale_q.io.enq.bits  := write_norm_q.io.deq.bits
  write_scale_q.io.deq.ready := false.B

  write_issue_q.io.enq.valid := false.B
  write_issue_q.io.enq.bits := write_scale_q.io.deq.bits

  // Garbage can immediately fire from dispatch_q -> norm_q
  when (write_dispatch_q.bits.laddr.is_garbage()) {
    write_norm_q.io.enq <> write_dispatch_q
  }

  // Non-acc or garbage can immediately fire between norm_q and scale_q
  when (write_norm_q.io.deq.bits.laddr.is_garbage() || !write_norm_q.io.deq.bits.laddr.is_acc_addr) {
    write_scale_q.io.enq <> write_norm_q.io.deq
  }

  // Non-acc or garbage can immediately fire between scale_q and issue_q
  when (write_scale_q.io.deq.bits.laddr.is_garbage() || !write_scale_q.io.deq.bits.laddr.is_acc_addr) {
    write_issue_q.io.enq <> write_scale_q.io.deq
  }

  val writeData = Wire(Valid(UInt((spad_w max acc_w).W)))
  writeData.valid := write_issue_q.io.deq.bits.laddr.is_garbage()
  writeData.bits := DontCare
  val fullAccWriteData = Wire(UInt(acc_w.W))
  fullAccWriteData := DontCare
  val writeData_is_full_width = !write_issue_q.io.deq.bits.laddr.is_garbage() &&
    write_issue_q.io.deq.bits.laddr.is_acc_addr && write_issue_q.io.deq.bits.laddr.read_full_acc_row
  val writeData_is_all_zeros = write_issue_q.io.deq.bits.laddr.is_garbage()

  writer.io.req.valid := write_issue_q.io.deq.valid && writeData.valid
  write_issue_q.io.deq.ready := writer.io.req.ready && writeData.valid
  writer.io.req.bits.vaddr := write_issue_q.io.deq.bits.vaddr
  writer.io.req.bits.len := Mux(writeData_is_full_width,
    write_issue_q.io.deq.bits.len * (accType.getWidth / 8).U,
    write_issue_q.io.deq.bits.len * (inputType.getWidth / 8).U)
  writer.io.req.bits.data := MuxCase(writeData.bits, Seq(
     writeData_is_all_zeros -> 0.U,
     writeData_is_full_width -> fullAccWriteData
  ))
  writer.io.req.bits.block := write_issue_q.io.deq.bits.block
  writer.io.req.bits.status := write_issue_q.io.deq.bits.status
  writer.io.req.bits.pool_en := write_issue_q.io.deq.bits.pool_en
  writer.io.req.bits.store_en := write_issue_q.io.deq.bits.store_en

  io.dma.write.resp.valid := false.B
  io.dma.write.resp.bits.cmd_id := write_dispatch_q.bits.cmd_id
  when (write_dispatch_q.bits.laddr.is_garbage() && write_dispatch_q.fire) {
    io.dma.write.resp.valid := true.B
  }

  read_issue_q.io.enq <> io.dma.read.req

  val zero_writer = Module(new ZeroWriter(config, new ScratchpadMemReadRequest(local_addr_t, mvin_scale_t_bits, coreMaxAddrBits)))

  when (io.dma.read.req.bits.all_zeros) {
    read_issue_q.io.enq.valid := false.B
    io.dma.read.req.ready := zero_writer.io.req.ready
  }

  zero_writer.io.req.valid := io.dma.read.req.valid && io.dma.read.req.bits.all_zeros
  zero_writer.io.req.bits.laddr := io.dma.read.req.bits.laddr
  zero_writer.io.req.bits.cols := io.dma.read.req.bits.cols
  zero_writer.io.req.bits.block_stride := io.dma.read.req.bits.block_stride
  zero_writer.io.req.bits.tag := io.dma.read.req.bits

  val zero_writer_pixel_repeater = Module(new PixelRepeater(inputType, local_addr_t, block_cols, aligned_to, new ScratchpadMemReadRequest(local_addr_t, mvin_scale_t_bits, coreMaxAddrBits), passthrough = !has_first_layer_optimizations))
  zero_writer_pixel_repeater.io.req.valid := zero_writer.io.resp.valid
  zero_writer_pixel_repeater.io.req.bits.in := 0.U.asTypeOf(Vec(block_cols, inputType))
  zero_writer_pixel_repeater.io.req.bits.laddr := zero_writer.io.resp.bits.laddr
  zero_writer_pixel_repeater.io.req.bits.len := zero_writer.io.resp.bits.tag.cols
  zero_writer_pixel_repeater.io.req.bits.pixel_repeats := zero_writer.io.resp.bits.tag.pixel_repeats
  zero_writer_pixel_repeater.io.req.bits.last := zero_writer.io.resp.bits.last
  zero_writer_pixel_repeater.io.req.bits.tag := zero_writer.io.resp.bits.tag
  zero_writer_pixel_repeater.io.req.bits.mask := {
    val n = inputType.getWidth / 8
    val mask = zero_writer.io.resp.bits.mask
    val expanded = VecInit(mask.flatMap(e => Seq.fill(n)(e)))
    expanded
  }

  zero_writer.io.resp.ready := zero_writer_pixel_repeater.io.req.ready
  zero_writer_pixel_repeater.io.resp.ready := false.B

  reader.io.req.valid := read_issue_q.io.deq.valid
  read_issue_q.io.deq.ready := reader.io.req.ready
  reader.io.req.bits.vaddr := read_issue_q.io.deq.bits.vaddr
  reader.io.req.bits.spaddr := Mux(read_issue_q.io.deq.bits.laddr.is_acc_addr,
    read_issue_q.io.deq.bits.laddr.full_acc_addr(), read_issue_q.io.deq.bits.laddr.full_sp_addr())
  reader.io.req.bits.len := read_issue_q.io.deq.bits.cols
  reader.io.req.bits.repeats := read_issue_q.io.deq.bits.repeats
  reader.io.req.bits.pixel_repeats := read_issue_q.io.deq.bits.pixel_repeats
  reader.io.req.bits.scale := read_issue_q.io.deq.bits.scale
  reader.io.req.bits.is_acc := read_issue_q.io.deq.bits.laddr.is_acc_addr
  reader.io.req.bits.accumulate := read_issue_q.io.deq.bits.laddr.accumulate
  reader.io.req.bits.has_acc_bitwidth := read_issue_q.io.deq.bits.has_acc_bitwidth
  reader.io.req.bits.block_stride := read_issue_q.io.deq.bits.block_stride
  reader.io.req.bits.status := read_issue_q.io.deq.bits.status
  reader.io.req.bits.cmd_id := read_issue_q.io.deq.bits.cmd_id

  val (mvin_scale_in, mvin_scale_out) = VectorScalarMultiplier(
    config.mvin_scale_args, config.inputType, config.meshColumns * config.tileColumns,
    chiselTypeOf(reader.io.resp.bits),
    is_acc = false
  )
  val (mvin_scale_acc_in, mvin_scale_acc_out) = if (mvin_scale_shared) (mvin_scale_in, mvin_scale_out) else (
    VectorScalarMultiplier(
      config.mvin_scale_acc_args, config.accType, config.meshColumns * config.tileColumns,
      chiselTypeOf(reader.io.resp.bits),
      is_acc = true
    )
  )

  mvin_scale_in.valid := reader.io.resp.valid && (mvin_scale_shared.B || !reader.io.resp.bits.is_acc ||
    (reader.io.resp.bits.is_acc && !reader.io.resp.bits.has_acc_bitwidth))

  mvin_scale_in.bits.in := reader.io.resp.bits.data.asTypeOf(chiselTypeOf(mvin_scale_in.bits.in))
  mvin_scale_in.bits.scale := reader.io.resp.bits.scale.asTypeOf(mvin_scale_t)
  mvin_scale_in.bits.repeats := reader.io.resp.bits.repeats
  mvin_scale_in.bits.pixel_repeats := reader.io.resp.bits.pixel_repeats
  mvin_scale_in.bits.last := reader.io.resp.bits.last
  mvin_scale_in.bits.tag := reader.io.resp.bits

  val mvin_scale_pixel_repeater = Module(new PixelRepeater(inputType, local_addr_t, block_cols, aligned_to, mvin_scale_out.bits.tag.cloneType, passthrough = !has_first_layer_optimizations))
  mvin_scale_pixel_repeater.io.req.valid := mvin_scale_out.valid
  mvin_scale_pixel_repeater.io.req.bits.in := mvin_scale_out.bits.out
  mvin_scale_pixel_repeater.io.req.bits.mask := mvin_scale_out.bits.tag.mask take mvin_scale_pixel_repeater.io.req.bits.mask.size
  mvin_scale_pixel_repeater.io.req.bits.laddr := mvin_scale_out.bits.tag.addr.asTypeOf(local_addr_t) + mvin_scale_out.bits.row
  mvin_scale_pixel_repeater.io.req.bits.len := mvin_scale_out.bits.tag.len
  mvin_scale_pixel_repeater.io.req.bits.pixel_repeats := mvin_scale_out.bits.tag.pixel_repeats
  mvin_scale_pixel_repeater.io.req.bits.last := mvin_scale_out.bits.last
  mvin_scale_pixel_repeater.io.req.bits.tag := mvin_scale_out.bits.tag

  mvin_scale_out.ready := mvin_scale_pixel_repeater.io.req.ready
  mvin_scale_pixel_repeater.io.resp.ready := false.B

  if (!mvin_scale_shared) {
    mvin_scale_acc_in.valid := reader.io.resp.valid &&
      (reader.io.resp.bits.is_acc && reader.io.resp.bits.has_acc_bitwidth)
    mvin_scale_acc_in.bits.in := reader.io.resp.bits.data.asTypeOf(chiselTypeOf(mvin_scale_acc_in.bits.in))
    mvin_scale_acc_in.bits.scale := reader.io.resp.bits.scale.asTypeOf(mvin_scale_acc_t)
    mvin_scale_acc_in.bits.repeats := reader.io.resp.bits.repeats
    mvin_scale_acc_in.bits.pixel_repeats := 1.U
    mvin_scale_acc_in.bits.last := reader.io.resp.bits.last
    mvin_scale_acc_in.bits.tag := reader.io.resp.bits

    mvin_scale_acc_out.ready := false.B
  }

  reader.io.resp.ready := Mux(reader.io.resp.bits.is_acc && reader.io.resp.bits.has_acc_bitwidth,
    mvin_scale_acc_in.ready, mvin_scale_in.ready)

  val mvin_scale_finished = mvin_scale_pixel_repeater.io.resp.fire && mvin_scale_pixel_repeater.io.resp.bits.last
  val mvin_scale_acc_finished = mvin_scale_acc_out.fire && mvin_scale_acc_out.bits.last
  val zero_writer_finished = zero_writer_pixel_repeater.io.resp.fire && zero_writer_pixel_repeater.io.resp.bits.last

  val zero_writer_bytes_read = Mux(zero_writer_pixel_repeater.io.resp.bits.laddr.is_acc_addr,
    zero_writer_pixel_repeater.io.resp.bits.tag.cols * (accType.getWidth / 8).U,
    zero_writer_pixel_repeater.io.resp.bits.tag.cols * (inputType.getWidth / 8).U)

  // For DMA read responses, mvin_scale gets first priority, then mvin_scale_acc, and then zero_writer
  io.dma.read.resp.valid := mvin_scale_finished || mvin_scale_acc_finished || zero_writer_finished

  io.dma.read.resp.bits.cmd_id := MuxCase(zero_writer_pixel_repeater.io.resp.bits.tag.cmd_id, Seq(
    mvin_scale_finished -> mvin_scale_pixel_repeater.io.resp.bits.tag.cmd_id,
    mvin_scale_acc_finished -> mvin_scale_acc_out.bits.tag.cmd_id))

  io.dma.read.resp.bits.bytesRead := MuxCase(zero_writer_bytes_read, Seq(
    mvin_scale_finished -> mvin_scale_pixel_repeater.io.resp.bits.tag.bytes_read,
    mvin_scale_acc_finished -> mvin_scale_acc_out.bits.tag.bytes_read))

  writer.io.flush := io.flush
  reader.io.flush := io.flush

  io.busy := writer.io.busy || reader.io.busy || write_issue_q.io.deq.valid || write_norm_q.io.deq.valid || write_scale_q.io.deq.valid || write_dispatch_q.valid

  val spad_mems = {
    val banks = Seq.fill(sp_banks) { Module(new ScratchpadBank(
      sp_bank_entries, spad_w,
      aligned_to, config.sp_singleported,
      use_shared_ext_mem, is_dummy, coreMaxAddrBits
    )) }
    val bank_ios = VecInit(banks.map(_.io))
    // Reading from the SRAM banks
    bank_ios.zipWithIndex.foreach { case (bio, i) =>
      if (use_shared_ext_mem) {
        io.ext_mem.get.spad(i) <> bio.ext_mem.get
      }

      val ex_read_req = io.srams.read(i).req
      val exread = ex_read_req.valid

      // TODO we tie the write dispatch queue's, and write issue queue's, ready and valid signals together here
      val dmawrite = write_dispatch_q.valid && write_norm_q.io.enq.ready &&
        !write_dispatch_q.bits.laddr.is_garbage() &&
        !(bio.write.en && config.sp_singleported.B) &&
        !write_dispatch_q.bits.laddr.is_acc_addr && write_dispatch_q.bits.laddr.sp_bank() === i.U

      bio.read.req.valid := exread || dmawrite
      ex_read_req.ready := bio.read.req.ready

      // The ExecuteController gets priority when reading from SRAMs
      when (exread) {
        bio.read.req.bits.addr := ex_read_req.bits.addr
        bio.read.req.bits.fromDMA := false.B
      }.elsewhen (dmawrite) {
        bio.read.req.bits.addr := write_dispatch_q.bits.laddr.sp_row()
        bio.read.req.bits.fromDMA := true.B

        when (bio.read.req.fire) {
          write_dispatch_q.ready := true.B
          write_norm_q.io.enq.valid := true.B

          io.dma.write.resp.valid := true.B
        }
      }.otherwise {
        bio.read.req.bits := DontCare
      }

      val dma_read_resp = Wire(Decoupled(new ScratchpadReadResp(spad_w)))
      dma_read_resp.valid := bio.read.resp.valid && bio.read.resp.bits.fromDMA
      dma_read_resp.bits := bio.read.resp.bits
      val ex_read_resp = Wire(Decoupled(new ScratchpadReadResp(spad_w)))
      ex_read_resp.valid := bio.read.resp.valid && !bio.read.resp.bits.fromDMA
      ex_read_resp.bits := bio.read.resp.bits

      val dma_read_pipe = Pipeline(dma_read_resp, spad_read_delay)
      val ex_read_pipe = Pipeline(ex_read_resp, spad_read_delay)

      bio.read.resp.ready := Mux(bio.read.resp.bits.fromDMA, dma_read_resp.ready, ex_read_resp.ready)

      dma_read_pipe.ready := writer.io.req.ready &&
        ((!write_issue_q.io.deq.bits.laddr.is_acc_addr && write_issue_q.io.deq.bits.laddr.sp_bank() === i.U) && write_issue_q.io.deq.valid) &&
        !write_issue_q.io.deq.bits.laddr.is_garbage()
      when (dma_read_pipe.fire) {
        writeData.valid := true.B
        writeData.bits := dma_read_pipe.bits.data
      }

      io.srams.read(i).resp <> ex_read_pipe
    }

    // Writing to the SRAM banks
    bank_ios.zipWithIndex.foreach { case (bio, i) =>
      val exwrite = io.srams.write(i).en

      val laddr = mvin_scale_pixel_repeater.io.resp.bits.laddr

      val dmaread = mvin_scale_pixel_repeater.io.resp.valid && !mvin_scale_pixel_repeater.io.resp.bits.tag.is_acc &&
        laddr.sp_bank() === i.U

      // We need to make sure that we don't try to return a dma read resp from both zero_writer and either mvin_scale
      // or mvin_acc_scale at the same time. The scalers always get priority in those cases
      val zerowrite = zero_writer_pixel_repeater.io.resp.valid && !zero_writer_pixel_repeater.io.resp.bits.laddr.is_acc_addr &&
        zero_writer_pixel_repeater.io.resp.bits.laddr.sp_bank() === i.U &&
        !((mvin_scale_pixel_repeater.io.resp.valid && mvin_scale_pixel_repeater.io.resp.bits.last) || (mvin_scale_acc_out.valid && mvin_scale_acc_out.bits.last))

      bio.write.en := exwrite || dmaread || zerowrite

      when (exwrite) {
        bio.write.addr := io.srams.write(i).addr
        bio.write.data := io.srams.write(i).data
        bio.write.mask := io.srams.write(i).mask
      }.elsewhen (dmaread) {
        bio.write.addr := laddr.sp_row()
        bio.write.data := mvin_scale_pixel_repeater.io.resp.bits.out.asUInt
        bio.write.mask := mvin_scale_pixel_repeater.io.resp.bits.mask take ((spad_w / (aligned_to * 8)) max 1)

        mvin_scale_pixel_repeater.io.resp.ready := true.B // TODO we combinationally couple valid and ready signals
      }.elsewhen (zerowrite) {
        bio.write.addr := zero_writer_pixel_repeater.io.resp.bits.laddr.sp_row()
        bio.write.data := 0.U
        bio.write.mask := zero_writer_pixel_repeater.io.resp.bits.mask

        zero_writer_pixel_repeater.io.resp.ready := true.B // TODO we combinationally couple valid and ready signals
      }.otherwise {
        bio.write.addr := DontCare
        bio.write.data := DontCare
        bio.write.mask := DontCare
      }
    }
    banks
  }

  val acc_row_t = Vec(meshColumns, Vec(tileColumns, accType))
  val spad_row_t = Vec(meshColumns, Vec(tileColumns, inputType))

  val (acc_norm_unit_in, acc_norm_unit_out) = Normalizer(
    is_passthru = !config.has_normalizations,
    max_len = block_cols,
    num_reduce_lanes = -1,
    num_stats = 2,
    latency = 4,
    fullDataType = acc_row_t,
    scale_t = acc_scale_t,
  )

  acc_norm_unit_in.valid := false.B
  acc_norm_unit_in.bits.len := write_norm_q.io.deq.bits.len
  acc_norm_unit_in.bits.stats_id := write_norm_q.io.deq.bits.acc_norm_stats_id
  acc_norm_unit_in.bits.cmd := write_norm_q.io.deq.bits.laddr.norm_cmd
  acc_norm_unit_in.bits.acc_read_resp := DontCare

  val acc_scale_unit = Module(new AccumulatorScale(
    acc_row_t,
    spad_row_t,
    acc_scale_t.asInstanceOf[V],
    acc_read_small_width,
    acc_read_full_width,
    acc_scale_func,
    acc_scale_num_units,
    acc_scale_latency,
    has_nonlinear_activations,
    has_normalizations,
  ))

  val acc_waiting_to_be_scaled = write_scale_q.io.deq.valid &&
    !write_scale_q.io.deq.bits.laddr.is_garbage() &&
    write_scale_q.io.deq.bits.laddr.is_acc_addr &&
    write_issue_q.io.enq.ready

  acc_norm_unit_out.ready := acc_scale_unit.io.in.ready && acc_waiting_to_be_scaled
  acc_scale_unit.io.in.valid := acc_norm_unit_out.valid && acc_waiting_to_be_scaled
  acc_scale_unit.io.in.bits  := acc_norm_unit_out.bits

  when (acc_scale_unit.io.in.fire) {
    write_issue_q.io.enq <> write_scale_q.io.deq
  }

  acc_scale_unit.io.out.ready := false.B

  val dma_resp_ready =
    writer.io.req.ready &&
      write_issue_q.io.deq.bits.laddr.is_acc_addr &&
      !write_issue_q.io.deq.bits.laddr.is_garbage()

  when (acc_scale_unit.io.out.bits.fromDMA && dma_resp_ready) {
    // Send the acc-scale result into the DMA
    acc_scale_unit.io.out.ready := true.B
    writeData.valid := acc_scale_unit.io.out.valid
    writeData.bits  := acc_scale_unit.io.out.bits.data.asUInt
    fullAccWriteData := acc_scale_unit.io.out.bits.full_data.asUInt
  }
  for (i <- 0 until acc_banks) {
    // Send the acc-sccale result to the ExController
    io.acc.read_resp(i).valid := false.B
    io.acc.read_resp(i).bits  := acc_scale_unit.io.out.bits
    when (!acc_scale_unit.io.out.bits.fromDMA && acc_scale_unit.io.out.bits.acc_bank_id === i.U) {
      acc_scale_unit.io.out.ready := io.acc.read_resp(i).ready
      io.acc.read_resp(i).valid := acc_scale_unit.io.out.valid
    }
  }

  val acc_adders = Module(new AccPipeShared(acc_latency-1, acc_row_t, acc_banks))

  val acc_mems = {
    val banks = Seq.fill(acc_banks) { Module(new AccumulatorMem(
      acc_bank_entries, acc_row_t, acc_scale_func, acc_scale_t.asInstanceOf[V],
      acc_singleported, acc_sub_banks,
      use_shared_ext_mem,
      acc_latency, accType, is_dummy
    )) }
    val bank_ios = VecInit(banks.map(_.io))

    // Getting the output of the bank that's about to be issued to the writer
    val bank_issued_io = bank_ios(write_issue_q.io.deq.bits.laddr.acc_bank())

    // Reading from the Accumulator banks
    bank_ios.zipWithIndex.foreach { case (bio, i) =>
      if (use_shared_ext_mem) {
        io.ext_mem.get.acc(i) <> bio.ext_mem.get
      }

      acc_adders.io.in_sel(i) := bio.adder.valid
      acc_adders.io.ina(i) := bio.adder.op1
      acc_adders.io.inb(i) := bio.adder.op2
      bio.adder.sum := acc_adders.io.out

      val ex_read_req = io.acc.read_req(i)
      val exread = ex_read_req.valid

      // TODO we tie the write dispatch queue's, and write issue queue's, ready and valid signals together here
      val dmawrite = write_dispatch_q.valid && write_norm_q.io.enq.ready &&
        !write_dispatch_q.bits.laddr.is_garbage() &&
        write_dispatch_q.bits.laddr.is_acc_addr && write_dispatch_q.bits.laddr.acc_bank() === i.U

      bio.read.req.valid := exread || dmawrite
      ex_read_req.ready := bio.read.req.ready

      // The ExecuteController gets priority when reading from accumulator banks
      when (exread) {
        bio.read.req.bits.addr := ex_read_req.bits.addr
        bio.read.req.bits.act := ex_read_req.bits.act
        bio.read.req.bits.igelu_qb := ex_read_req.bits.igelu_qb
        bio.read.req.bits.igelu_qc := ex_read_req.bits.igelu_qc
        bio.read.req.bits.iexp_qln2 := ex_read_req.bits.iexp_qln2
        bio.read.req.bits.iexp_qln2_inv := ex_read_req.bits.iexp_qln2_inv
        bio.read.req.bits.scale := ex_read_req.bits.scale
        bio.read.req.bits.full := false.B
        bio.read.req.bits.fromDMA := false.B
      }.elsewhen (dmawrite) {
        bio.read.req.bits.addr := write_dispatch_q.bits.laddr.acc_row()
        bio.read.req.bits.full := write_dispatch_q.bits.laddr.read_full_acc_row
        bio.read.req.bits.act := write_dispatch_q.bits.acc_act
        bio.read.req.bits.igelu_qb := write_dispatch_q.bits.acc_igelu_qb.asTypeOf(bio.read.req.bits.igelu_qb)
        bio.read.req.bits.igelu_qc := write_dispatch_q.bits.acc_igelu_qc.asTypeOf(bio.read.req.bits.igelu_qc)
        bio.read.req.bits.iexp_qln2 := write_dispatch_q.bits.acc_iexp_qln2.asTypeOf(bio.read.req.bits.iexp_qln2)
        bio.read.req.bits.iexp_qln2_inv := write_dispatch_q.bits.acc_iexp_qln2_inv.asTypeOf(bio.read.req.bits.iexp_qln2_inv)
        bio.read.req.bits.scale := write_dispatch_q.bits.acc_scale.asTypeOf(bio.read.req.bits.scale)
        bio.read.req.bits.fromDMA := true.B

        when (bio.read.req.fire) {
          write_dispatch_q.ready := true.B
          write_norm_q.io.enq.valid := true.B

          io.dma.write.resp.valid := true.B
        }
      }.otherwise {
        bio.read.req.bits := DontCare
      }
      bio.read.resp.ready := false.B

      when (write_norm_q.io.deq.valid &&
        acc_norm_unit_in.ready &&
        bio.read.resp.valid &&
        write_scale_q.io.enq.ready &&
        write_norm_q.io.deq.bits.laddr.is_acc_addr &&
        !write_norm_q.io.deq.bits.laddr.is_garbage() &&
        write_norm_q.io.deq.bits.laddr.acc_bank() === i.U)
      {
        write_norm_q.io.deq.ready := true.B
        acc_norm_unit_in.valid := true.B
        bio.read.resp.ready := true.B

        // Some normalizer commands don't write to main memory, so they don't need to be passed on to the scaling units
        write_scale_q.io.enq.valid := NormCmd.writes_to_main_memory(write_norm_q.io.deq.bits.laddr.norm_cmd)

        acc_norm_unit_in.bits.acc_read_resp := bio.read.resp.bits
        acc_norm_unit_in.bits.acc_read_resp.acc_bank_id := i.U
      }
    }

    // Writing to the accumulator banks
    bank_ios.zipWithIndex.foreach { case (bio, i) =>
      // Order of precedence during writes is ExecuteController, and then mvin_scale, and then mvin_scale_acc, and
      // then zero_writer

      val exwrite = io.acc.write(i).valid
      io.acc.write(i).ready := true.B
      assert(!(exwrite && !bio.write.ready), "Execute controller write to AccumulatorMem was skipped")

      val from_mvin_scale = mvin_scale_pixel_repeater.io.resp.valid && mvin_scale_pixel_repeater.io.resp.bits.tag.is_acc
      val from_mvin_scale_acc = mvin_scale_acc_out.valid && mvin_scale_acc_out.bits.tag.is_acc

      val mvin_scale_laddr = mvin_scale_pixel_repeater.io.resp.bits.laddr
      val mvin_scale_acc_laddr = mvin_scale_acc_out.bits.tag.addr.asTypeOf(local_addr_t) + mvin_scale_acc_out.bits.row

      val dmaread_bank = Mux(from_mvin_scale, mvin_scale_laddr.acc_bank(),
        mvin_scale_acc_laddr.acc_bank())
      val dmaread_row = Mux(from_mvin_scale, mvin_scale_laddr.acc_row(), mvin_scale_acc_laddr.acc_row())

      // We need to make sure that we don't try to return a dma read resp from both mvin_scale and mvin_scale_acc
      // at the same time. mvin_scale always gets priority in this cases
      val spad_last = mvin_scale_pixel_repeater.io.resp.valid && mvin_scale_pixel_repeater.io.resp.bits.last && !mvin_scale_pixel_repeater.io.resp.bits.tag.is_acc

      val dmaread = (from_mvin_scale || from_mvin_scale_acc) &&
        dmaread_bank === i.U

      // We need to make sure that we don't try to return a dma read resp from both zero_writer and either mvin_scale
      // or mvin_acc_scale at the same time. The scalers always get priority in those cases
      val zerowrite = zero_writer_pixel_repeater.io.resp.valid && zero_writer_pixel_repeater.io.resp.bits.laddr.is_acc_addr &&
        zero_writer_pixel_repeater.io.resp.bits.laddr.acc_bank() === i.U &&
        !((mvin_scale_pixel_repeater.io.resp.valid && mvin_scale_pixel_repeater.io.resp.bits.last) || (mvin_scale_acc_out.valid && mvin_scale_acc_out.bits.last))

      val consecutive_write_block = RegInit(false.B)
      if (acc_singleported) {
        val consecutive_write_sub_bank = RegInit(0.U((1 max log2Ceil(acc_sub_banks)).W))
        when (bio.write.fire && bio.write.bits.acc &&
          (bio.write.bits.addr(log2Ceil(acc_sub_banks)-1,0) === consecutive_write_sub_bank)) {
          consecutive_write_block := true.B
        } .elsewhen (bio.write.fire && bio.write.bits.acc) {
          consecutive_write_block := false.B
          consecutive_write_sub_bank := bio.write.bits.addr(log2Ceil(acc_sub_banks)-1,0)
        } .otherwise {
          consecutive_write_block := false.B
        }
      }
      bio.write.valid := false.B

      bio.write.bits.acc := MuxCase(zero_writer_pixel_repeater.io.resp.bits.laddr.accumulate,
        Seq(exwrite -> io.acc.write(i).bits.acc,
          from_mvin_scale -> mvin_scale_pixel_repeater.io.resp.bits.tag.accumulate,
          from_mvin_scale_acc -> mvin_scale_acc_out.bits.tag.accumulate))

      bio.write.bits.addr := MuxCase(zero_writer_pixel_repeater.io.resp.bits.laddr.acc_row(),
        Seq(exwrite -> io.acc.write(i).bits.addr,
          (from_mvin_scale || from_mvin_scale_acc) -> dmaread_row))

      when (exwrite) {
        bio.write.valid := true.B
        bio.write.bits.data := io.acc.write(i).bits.data
        bio.write.bits.mask := io.acc.write(i).bits.mask
      }.elsewhen (dmaread && !spad_last && !consecutive_write_block) {
        bio.write.valid := true.B
        bio.write.bits.data := Mux(from_mvin_scale,
          VecInit(mvin_scale_pixel_repeater.io.resp.bits.out.map(e => e.withWidthOf(accType))).asTypeOf(acc_row_t),
          mvin_scale_acc_out.bits.out.asTypeOf(acc_row_t))
        bio.write.bits.mask :=
          Mux(from_mvin_scale,
            {
              val n = accType.getWidth / inputType.getWidth
              val mask = mvin_scale_pixel_repeater.io.resp.bits.mask take ((spad_w / (aligned_to * 8)) max 1)
              val expanded = VecInit(mask.flatMap(e => Seq.fill(n)(e)))
              expanded
            },
            mvin_scale_acc_out.bits.tag.mask)

        when(from_mvin_scale) {
          mvin_scale_pixel_repeater.io.resp.ready := bio.write.ready
        }.otherwise {
          mvin_scale_acc_out.ready := bio.write.ready
        }
      }.elsewhen (zerowrite && !spad_last && !consecutive_write_block) {
        bio.write.valid := true.B
        bio.write.bits.data := 0.U.asTypeOf(acc_row_t)
        bio.write.bits.mask := {
          val n = accType.getWidth / inputType.getWidth
          val mask = zero_writer_pixel_repeater.io.resp.bits.mask
          val expanded = VecInit(mask.flatMap(e => Seq.fill(n)(e)))
          expanded
        }

        zero_writer_pixel_repeater.io.resp.ready := bio.write.ready
      }.otherwise {
        bio.write.bits.data := DontCare
        bio.write.bits.mask := DontCare
      }
    }
    banks
  }

  // Counter connection
  io.counter := DontCare
  io.counter.collect(reader.io.counter)
  io.counter.collect(writer.io.counter)
}

class StreamReadRequest[U <: Data](spad_rows: Int, acc_rows: Int, mvin_scale_t_bits: Int, coreMaxAddrBits: Int) extends Bundle {
  val vaddr = UInt(coreMaxAddrBits.W)
  val spaddr = UInt(log2Up(spad_rows max acc_rows).W) // TODO use LocalAddr in DMA
  val is_acc = Bool()
  val accumulate = Bool()
  val has_acc_bitwidth = Bool()
  val scale = UInt(mvin_scale_t_bits.W)
  val status = new MStatus
  val len = UInt(16.W) // TODO magic number
  val repeats = UInt(16.W) // TODO magic number
  val pixel_repeats = UInt(8.W) // TODO magic number
  val block_stride = UInt(16.W) // TODO magic number
  val cmd_id = UInt(8.W) // TODO magic number

}

class StreamReadResponse[U <: Data](spadWidth: Int, accWidth: Int, spad_rows: Int, acc_rows: Int, aligned_to: Int, mvin_scale_t_bits: Int, coreMaxAddrBits: Int)
                         extends Bundle {
  val data = UInt((spadWidth max accWidth).W)
  val addr = UInt(log2Up(spad_rows max acc_rows).W)
  val mask = Vec((spadWidth max accWidth) / (aligned_to * 8) max 1, Bool())
  val is_acc = Bool()
  val accumulate = Bool()
  val has_acc_bitwidth = Bool()
  val scale = UInt(mvin_scale_t_bits.W)
  val repeats = UInt(16.W) // TODO magic number
  val pixel_repeats = UInt(16.W) // TODO magic number
  val len = UInt(16.W) // TODO magic number
  val last = Bool()
  val bytes_read = UInt(8.W) // TODO magic number
  val cmd_id = UInt(8.W) // TODO magic number

}

//FIXME: coreMaxAddrBits == vaddrBits when is this not true?
class StreamReader[T <: Data, U <: Data, V <: Data](config: GemminiArrayConfig[T, U, V], nXacts: Int, beatBits: Int, maxBytes: Int, spadWidth: Int, accWidth: Int, aligned_to: Int,
                   spad_rows: Int, acc_rows: Int, meshRows: Int, use_tlb_register_filter: Boolean,
                   coreMaxAddrBits: Int, vaddrBits: Int) extends Module {

  val core = Module(new StreamReaderCore(config, nXacts, beatBits, maxBytes, spadWidth, accWidth, aligned_to, spad_rows, acc_rows, meshRows, use_tlb_register_filter, coreMaxAddrBits, vaddrBits))


  val io = IO(new Bundle {
    val req = Flipped(Decoupled(new StreamReadRequest(spad_rows, acc_rows, config.mvin_scale_t_bits, coreMaxAddrBits)))
    val resp = Decoupled(new StreamReadResponse(spadWidth, accWidth, spad_rows, acc_rows, aligned_to, config.mvin_scale_t_bits, coreMaxAddrBits))
    val busy = Output(Bool())
    val flush = Input(Bool())

    val reqPacket = Decoupled(new ReadReqPacket(maxBytes, nXacts, vaddrBits))
    val respPacket = Flipped(Decoupled(new ReadRespPacket(maxBytes, nXacts, beatBits)))

    val counter = new CounterEventIO()
  })

  val nCmds = (nXacts / meshRows) + 1

  val xactTracker = Module(new XactTracker(nXacts, maxBytes, spadWidth, accWidth, spad_rows, acc_rows, maxBytes, config.mvin_scale_t_bits, nCmds))

  val beatPacker = Module(new BeatMerger(beatBits, maxBytes, spadWidth, accWidth, spad_rows, acc_rows, maxBytes, aligned_to, meshRows, config.mvin_scale_t_bits, nCmds))

  core.io.req <> io.req
  core.io.reqPacket <> io.reqPacket
  core.io.respPacket <> io.respPacket

  io.busy := xactTracker.io.busy
  core.io.flush := io.flush

  xactTracker.io.alloc <> core.io.reserve
  xactTracker.io.peek.xactid := RegEnableThru(core.io.beatData.bits.xactid, beatPacker.io.req.fire)
  xactTracker.io.peek.pop := beatPacker.io.in.fire && core.io.beatData.bits.last

  core.io.beatData.ready := beatPacker.io.in.ready
  beatPacker.io.req.valid := core.io.beatData.valid
  beatPacker.io.req.bits := xactTracker.io.peek.entry
  beatPacker.io.req.bits.lg_len_req := core.io.beatData.bits.lg_len_req
  beatPacker.io.in.valid := core.io.beatData.valid
  beatPacker.io.in.bits := core.io.beatData.bits.data

  beatPacker.io.out.ready := io.resp.ready
  io.resp.valid := beatPacker.io.out.valid
  io.resp.bits.data := beatPacker.io.out.bits.data
  io.resp.bits.addr := beatPacker.io.out.bits.addr
  io.resp.bits.mask := beatPacker.io.out.bits.mask
  io.resp.bits.is_acc := beatPacker.io.out.bits.is_acc
  io.resp.bits.accumulate := beatPacker.io.out.bits.accumulate
  io.resp.bits.has_acc_bitwidth := beatPacker.io.out.bits.has_acc_bitwidth
  io.resp.bits.scale := RegEnable(xactTracker.io.peek.entry.scale, beatPacker.io.req.fire)
  io.resp.bits.repeats := RegEnable(xactTracker.io.peek.entry.repeats, beatPacker.io.req.fire)
  io.resp.bits.pixel_repeats := RegEnable(xactTracker.io.peek.entry.pixel_repeats, beatPacker.io.req.fire)
  io.resp.bits.len := RegEnable(xactTracker.io.peek.entry.len, beatPacker.io.req.fire)
  io.resp.bits.cmd_id := RegEnable(xactTracker.io.peek.entry.cmd_id, beatPacker.io.req.fire)
  io.resp.bits.bytes_read := RegEnable(xactTracker.io.peek.entry.bytes_to_read, beatPacker.io.req.fire)
  io.resp.bits.last := beatPacker.io.out.bits.last

  io.counter := DontCare
  io.counter.collect(core.io.counter)
  io.counter.collect(xactTracker.io.counter)
}

class StreamReadBeat (val nXacts: Int, val beatBits: Int, val maxReqBytes: Int) extends Bundle {
  val xactid = UInt(log2Up(nXacts).W)
  val data = UInt(beatBits.W)
  val lg_len_req = UInt(log2Up(log2Up(maxReqBytes+1)+1).W)
  val last = Bool()
}

class StreamReaderCore[T <: Data, U <: Data, V <: Data](config: GemminiArrayConfig[T, U, V], nXacts: Int, beatBits: Int,
                                                        maxBytes: Int, spadWidth: Int, accWidth: Int, aligned_to: Int,
                                                        spad_rows: Int, acc_rows: Int, meshRows: Int,
                                                        use_tlb_register_filter: Boolean,
                                                        coreMaxAddrBits: Int, vaddrBits: Int) extends Module {

  require(isPow2(aligned_to))

  import config._

  // TODO when we request data from multiple rows which are actually contiguous in main memory, we should merge them into fewer requests

  val spadWidthBytes = spadWidth / 8
  val accWidthBytes = accWidth / 8
  val beatBytes = beatBits / 8

  val nCmds = (nXacts / meshRows) + 1

  val io = IO(new Bundle {
    val req = Flipped(Decoupled(new StreamReadRequest(spad_rows, acc_rows, config.mvin_scale_t_bits, coreMaxAddrBits)))

    val reserve = new XactTrackerAllocIO(nXacts, maxBytes, spadWidth, accWidth, spad_rows, acc_rows, maxBytes, config.mvin_scale_t_bits, nCmds)

    val beatData = Decoupled(new StreamReadBeat(nXacts, beatBits, maxBytes))

    val reqPacket = Decoupled(new ReadReqPacket(maxBytes, nXacts, vaddrBits))
    val respPacket = Flipped(Decoupled(new ReadRespPacket(maxBytes, nXacts, beatBits)))

    val flush = Input(Bool())
    val counter = new CounterEventIO()
  })

  val s_idle :: s_req_new_block :: Nil = Enum(2)
  val state = RegInit(s_idle)

  val req = Reg(new StreamReadRequest(spad_rows, acc_rows, config.mvin_scale_t_bits, coreMaxAddrBits))
  val vaddr = req.vaddr

  val bytesRequested = Reg(UInt(log2Ceil(spadWidthBytes max accWidthBytes max maxBytes).W)) // TODO this only needs to count up to (dataBytes/aligned_to), right?
  val bytesLeft = Mux(req.has_acc_bitwidth, req.len * (config.accType.getWidth / 8).U, req.len * (config.inputType.getWidth / 8).U) - bytesRequested

  val state_machine_ready_for_req = WireInit(state === s_idle)
  io.req.ready := state_machine_ready_for_req

  // TODO Can we filter out the larger read_sizes here if the systolic array is small, in the same way that we do so
  // for the write_sizes down below?
  val read_sizes = ((aligned_to max beatBytes) to maxBytes by aligned_to).
    filter(s => isPow2(s)).
    filter(s => s % beatBytes == 0)
  val read_packets = read_sizes.map { s =>
    val lg_s = log2Ceil(s)
    val vaddr_aligned_to_size = if (s == 1) vaddr else Cat(vaddr(vaddrBits-1, lg_s), 0.U(lg_s.W))
    val vaddr_offset = if (s > 1) vaddr(lg_s-1, 0) else 0.U

    val packet = Wire(new ReadReqPacket(maxBytes, nXacts, vaddrBits))
    packet.lg_size := lg_s.U
    packet.bytes_read := minOf(s.U - vaddr_offset, bytesLeft)
    packet.shift := vaddr_offset
    packet.vaddr := vaddr_aligned_to_size
    packet.source := DontCare

    packet
  }
  val read_packet = read_packets.reduce { (acc, p) =>
    Mux(p.bytes_read > acc.bytes_read, p, acc)
  }
  val read_vaddr = read_packet.vaddr
  val read_lg_size = read_packet.lg_size
  val read_bytes_read = read_packet.bytes_read
  val read_shift = read_packet.shift

  // Firing off read requests and allocating space inside the reservation buffer for them
  io.reqPacket.valid := state === s_req_new_block && io.reserve.ready
  io.reqPacket.bits.vaddr := read_vaddr
  io.reqPacket.bits.lg_size := read_lg_size
  io.reqPacket.bits.bytes_read := read_bytes_read
  io.reqPacket.bits.shift := read_shift
  io.reqPacket.bits.source := io.reserve.xactid

  io.reserve.valid := state === s_req_new_block && io.reqPacket.ready
  io.reserve.entry.shift := read_shift
  io.reserve.entry.is_acc := req.is_acc
  io.reserve.entry.accumulate := req.accumulate
  io.reserve.entry.has_acc_bitwidth := req.has_acc_bitwidth
  io.reserve.entry.scale := req.scale
  io.reserve.entry.repeats := req.repeats
  io.reserve.entry.pixel_repeats := req.pixel_repeats
  io.reserve.entry.len := req.len
  io.reserve.entry.block_stride := req.block_stride
  io.reserve.entry.lg_len_req := DontCare // TODO just remove this from the IO completely
  io.reserve.entry.bytes_to_read := read_bytes_read
  io.reserve.entry.cmd_id := req.cmd_id

  io.reserve.entry.addr := req.spaddr + req.block_stride *
    Mux(req.has_acc_bitwidth,
      // We only add "if" statements here to satisfy the Verilator linter. The code would be cleaner without the
      // "if" condition and the "else" clause. Similarly, the width expansions are also there to satisfy the Verilator
      // linter, despite making the code uglier.
      if (bytesRequested.getWidth >= log2Up(accWidthBytes+1)) bytesRequested / accWidthBytes.U(bytesRequested.getWidth.W) else 0.U,
      if (bytesRequested.getWidth >= log2Up(spadWidthBytes+1)) bytesRequested / spadWidthBytes.U(bytesRequested.getWidth.W) else 0.U)
  io.reserve.entry.spad_row_offset := Mux(req.has_acc_bitwidth, bytesRequested % accWidthBytes.U, bytesRequested % spadWidthBytes.U)

  when (io.reqPacket.fire) {
    val next_vaddr = req.vaddr + read_bytes_read
    req.vaddr := next_vaddr

    bytesRequested := bytesRequested + read_bytes_read

    when (read_bytes_read >= bytesLeft) {
      // We're done with this request at this point
      state_machine_ready_for_req := true.B
      state := s_idle
    }
  }

  // Forward read responses to the reservation buffer
  io.respPacket.ready := io.beatData.ready

  io.beatData.valid := io.respPacket.valid
  io.beatData.bits.xactid := io.respPacket.bits.source
  io.beatData.bits.data := io.respPacket.bits.data
  io.beatData.bits.lg_len_req := io.respPacket.bits.lg_size
  io.beatData.bits.last := io.respPacket.bits.last

  // Accepting requests to kick-start the state machine
  when (io.req.fire) {
    req := io.req.bits
    bytesRequested := 0.U

    state := s_req_new_block
  }

  // Performance counter
  io.counter := DontCare
  CounterEventIO.init(io.counter)
  io.counter.connectEventSignal(CounterEvent.RDMA_ACTIVE_CYCLE, state =/= s_idle)
  io.counter.connectEventSignal(CounterEvent.RDMA_TL_WAIT_CYCLES, io.reqPacket.valid && !io.reqPacket.ready)

  // External counters
  val total_bytes_read = RegInit(0.U(CounterExternal.EXTERNAL_WIDTH.W))
  when (io.counter.external_reset) {
    total_bytes_read := 0.U
  }.elsewhen (io.respPacket.fire) {
    total_bytes_read := total_bytes_read + (1.U << io.respPacket.bits.lg_size)
  }

  io.counter.connectExternalCounter(CounterExternal.RDMA_BYTES_REC, total_bytes_read)
}

class StreamWriteRequest(val dataWidth: Int, val maxBytes: Int, coreMaxAddrBits: Int) extends Bundle {
  val vaddr = UInt(coreMaxAddrBits.W)
  val data = UInt(dataWidth.W)
  val len = UInt(log2Up((dataWidth/8 max maxBytes)+1).W) // The number of bytes to write
  val block = UInt(8.W) // TODO magic number
  val status = new MStatus

  // Pooling variables
  val pool_en = Bool()
  val store_en = Bool()
}

class StreamWriter[T <: Data: Arithmetic](nXacts: Int, beatBits: Int, maxBytes: Int, dataWidth: Int, aligned_to: Int,
                                          inputType: T, block_cols: Int, use_tlb_register_filter: Boolean,
                                          coreMaxAddrBits: Int, vaddrBits: Int) extends Module {

  require(isPow2(aligned_to))

  val dataBytes = dataWidth / 8
  val beatBytes = beatBits / 8
  val lgBeatBytes = log2Ceil(beatBytes)
  val maxBeatsPerReq = maxBytes / beatBytes
  val inputTypeRowBytes = block_cols * inputType.getWidth / 8
  val maxBlocks = maxBytes / inputTypeRowBytes

  require(beatBytes > 0)

  val io = IO(new Bundle {
    val req = Flipped(Decoupled(new StreamWriteRequest(dataWidth, maxBytes, coreMaxAddrBits)))
    val busy = Output(Bool())
    val flush = Input(Bool())
    val counter = new CounterEventIO()

    val reqPacket = Decoupled(new WriteReqPacket(dataWidth, vaddrBits))
    val respPacket = Flipped(Decoupled(new WriteRespPacket(nXacts, maxBytes)))
  })

  val (s_idle :: s_writing_new_block :: s_writing_beats :: Nil) = Enum(3)
  val state = RegInit(s_idle)

  val req = Reg(new StreamWriteRequest(dataWidth, maxBytes, coreMaxAddrBits))

  // TODO use the same register to hold data_blocks and data_single_block, so that this Mux here is not necessary
  val data_blocks = Reg(Vec(maxBlocks, UInt((inputTypeRowBytes * 8).W)))
  val data_single_block = Reg(UInt(dataWidth.W)) // For data that's just one-block-wide
  val data = Mux(req.block === 0.U, data_single_block, data_blocks.asUInt)

  val bytesSent = Reg(UInt(log2Ceil((dataBytes max maxBytes)+1).W))  // TODO this only needs to count up to (dataBytes/aligned_to), right?
  val bytesLeft = req.len - bytesSent

  val xactBusy = RegInit(0.U(nXacts.W))
  val xactOnehot = PriorityEncoderOH(~xactBusy)
  val xactId = OHToUInt(xactOnehot)

  val xactBusy_fire = WireInit(false.B)
  val xactBusy_add = Mux(xactBusy_fire, (1.U << xactId).asUInt, 0.U)
  val xactBusy_remove = ~Mux(io.respPacket.fire, (1.U << io.respPacket.bits.source).asUInt, 0.U)
  xactBusy := (xactBusy | xactBusy_add) & xactBusy_remove.asUInt

  val state_machine_ready_for_req = WireInit(state === s_idle)
  io.req.ready := state_machine_ready_for_req
  io.busy := xactBusy.orR || (state =/= s_idle)

  val vaddr = req.vaddr

  // Select the size and mask of the write request
  class Packet extends Bundle {
    val size = UInt(log2Ceil(maxBytes+1).W)
    val lg_size = UInt(log2Ceil(log2Ceil(maxBytes+1)+1).W)
    val mask = Vec(maxBeatsPerReq, Vec(beatBytes, Bool()))
    val vaddr = UInt(vaddrBits.W)
    val is_full = Bool()

    val bytes_written = UInt(log2Up(maxBytes+1).W)
    val bytes_written_per_beat = Vec(maxBeatsPerReq, UInt(log2Up(beatBytes+1).W))

    def total_beats(dummy: Int = 0) = Mux(size < beatBytes.U, 1.U, size / beatBytes.U(size.getWidth.W)) // The width expansion is added here solely to satsify Verilator's linter
  }

  val smallest_write_size = aligned_to max beatBytes
  val write_sizes = (smallest_write_size to maxBytes by aligned_to).
    filter(s => isPow2(s)).
    filter(s => s % beatBytes == 0)
  val write_packets = write_sizes.map { s =>
    val lg_s = log2Ceil(s)
    val vaddr_aligned_to_size = if (s == 1) vaddr else Cat(vaddr(vaddrBits-1, lg_s), 0.U(lg_s.W))
    val vaddr_offset = if (s > 1) vaddr(lg_s - 1, 0) else 0.U

    val mask = (0 until maxBytes).map { i => i.U >= vaddr_offset && i.U < vaddr_offset +& bytesLeft && (i < s).B }

    val bytes_written = {
      Mux(vaddr_offset +& bytesLeft > s.U, s.U - vaddr_offset, bytesLeft)
    }

    val packet = Wire(new Packet())
    packet.size := s.U
    packet.lg_size := lg_s.U
    packet.mask := VecInit(mask.grouped(beatBytes).map(v => VecInit(v)).toSeq)
    packet.vaddr := vaddr_aligned_to_size
    packet.is_full := mask.take(s).reduce(_ && _)

    packet.bytes_written := bytes_written
    packet.bytes_written_per_beat.zipWithIndex.foreach { case (b, i) =>
      val start_of_beat = i * beatBytes
      val end_of_beat = (i+1) * beatBytes

      val left_shift = Mux(vaddr_offset >= start_of_beat.U && vaddr_offset < end_of_beat.U,
        vaddr_offset - start_of_beat.U,
        0.U)

      val right_shift = Mux(vaddr_offset +& bytesLeft >= start_of_beat.U && vaddr_offset +& bytesLeft < end_of_beat.U,
        end_of_beat.U - (vaddr_offset +& bytesLeft),
        0.U)

      val too_early = vaddr_offset >= end_of_beat.U
      val too_late = vaddr_offset +& bytesLeft <= start_of_beat.U

      b := Mux(too_early || too_late, 0.U, beatBytes.U - (left_shift +& right_shift))
    }

    packet
  }
  val best_write_packet = write_packets.reduce { (acc, p) =>
    Mux(p.bytes_written > acc.bytes_written, p, acc)
  }
  val write_packet = RegEnableThru(best_write_packet, state === s_writing_new_block)

  val write_size = write_packet.size
  val lg_write_size = write_packet.lg_size
  val write_beats = write_packet.total_beats()
  val write_vaddr = write_packet.vaddr
  val write_full = write_packet.is_full

  val beatsLeft = Reg(UInt(log2Up(maxBytes/aligned_to).W))
  val beatsSent = Mux(state === s_writing_new_block, 0.U, write_beats - beatsLeft)

  val write_mask = write_packet.mask(beatsSent)
  val write_shift = PriorityEncoder(write_mask)

  val bytes_written_this_beat = write_packet.bytes_written_per_beat(beatsSent)

  // Firing off write requests
  io.reqPacket.valid := (state === s_writing_new_block || state === s_writing_beats) && !xactBusy.andR

  io.reqPacket.bits.data := Mux(write_full,
    (data >> (bytesSent * 8.U)).asUInt,
    ((data >> (bytesSent * 8.U)) << (write_shift * 8.U)).asUInt)
  io.reqPacket.bits.vaddr := write_vaddr
  io.reqPacket.bits.last := Mux(state === s_writing_new_block, write_beats === 1.U, beatsLeft === 1.U)

  io.respPacket.ready := true.B

  when (io.reqPacket.fire) {
    when (state === s_writing_new_block) {
      beatsLeft := write_beats - 1.U

      val next_vaddr = req.vaddr + write_packet.bytes_written
      req.vaddr := next_vaddr

      bytesSent := bytesSent + bytes_written_this_beat

      when (write_beats === 1.U) {
        when (bytes_written_this_beat >= bytesLeft) {
          // We're done with this request at this point
          state_machine_ready_for_req := true.B
          state := s_idle
        }
      }.otherwise {
        state := s_writing_beats
      }
    }.elsewhen(state === s_writing_beats) {
      beatsLeft := beatsLeft - 1.U
      bytesSent := bytesSent + bytes_written_this_beat

      assert(beatsLeft > 0.U)

      when (beatsLeft === 1.U) {
        when (bytes_written_this_beat >= bytesLeft) {
          // We're done with this request at this point
          state_machine_ready_for_req := true.B
          state := s_idle
        }.otherwise {
          state := s_writing_new_block
        }
      }
    }
  }

  // Accepting requests to kick-start the state machine
  when (io.req.fire) {
    val pooled = {
      val cols = dataWidth / inputType.getWidth
      val v1 = io.req.bits.data.asTypeOf(Vec(cols, inputType))
      val v2 = data_single_block.asTypeOf(Vec(cols, inputType))
      val m = v1.zip(v2)
      VecInit(m.zipWithIndex.map{case ((x, y), i) => if (i < block_cols) maxOf(x, y) else y}).asUInt
    }

    req := io.req.bits
    req.len := io.req.bits.block * inputTypeRowBytes.U + io.req.bits.len

    data_single_block := Mux(io.req.bits.pool_en, pooled, io.req.bits.data)
    data_blocks(io.req.bits.block) := io.req.bits.data

    bytesSent := 0.U

    state := Mux(io.req.bits.store_en, s_writing_new_block, s_idle)

    assert(io.req.bits.len <= (block_cols * inputType.getWidth / 8).U || io.req.bits.block === 0.U, "DMA can't write multiple blocks to main memory when writing full accumulator output")
    assert(!io.req.bits.pool_en || io.req.bits.block === 0.U, "Can't pool with block-mvout")
  }

  // Performance counter
  io.counter := DontCare
  CounterEventIO.init(io.counter)
  io.counter.connectEventSignal(CounterEvent.WDMA_ACTIVE_CYCLE, state =/= s_idle)
  io.counter.connectEventSignal(CounterEvent.WDMA_TL_WAIT_CYCLES, io.reqPacket.valid && !io.reqPacket.ready)

  // External counters
  val total_bytes_sent = RegInit(0.U(CounterExternal.EXTERNAL_WIDTH.W))
  when (io.respPacket.fire) {
    total_bytes_sent := total_bytes_sent + (1.U << io.respPacket.bits.lg_size)
  }

  val total_latency = RegInit(0.U(CounterExternal.EXTERNAL_WIDTH.W))
  total_latency := total_latency + PopCount(xactBusy)

  when (io.counter.external_reset) {
    total_bytes_sent := 0.U
    total_latency := 0.U
  }

  io.counter.connectExternalCounter(CounterExternal.WDMA_BYTES_SENT, total_bytes_sent)
  io.counter.connectExternalCounter(CounterExternal.WDMA_TOTAL_LATENCY, total_latency)
}
