import chisel3._
import chisel3.util._

class RoCCInstruction extends Bundle {
  val funct = Bits(7.W)
  val rs2 = Bits(5.W)
  val rs1 = Bits(5.W)
  val xd = Bool()
  val xs1 = Bool()
  val xs2 = Bool()
  val rd = Bits(5.W)
  val opcode = Bits(7.W)
}

class RoCCCommand(xLen: Int) extends Bundle {
  val inst = new RoCCInstruction
  val rs1 = Bits(xLen.W)
  val rs2 = Bits(xLen.W)
  val status = new MStatus
}

class RoCCResponse(xLen: Int) extends Bundle {
  val rd = Bits(5.W)
  val data = Bits(xLen.W)
}

class RoCCCoreIO(val nRoCCCSRs: Int = 0, xLen: Int) extends Bundle {
  val cmd = Flipped(Decoupled(new RoCCCommand(xLen)))
  val resp = Decoupled(new RoCCResponse(xLen))
  // val mem = new HellaCacheIO
  val busy = Output(Bool())
  val interrupt = Output(Bool())
  val exception = Input(Bool())
  // val csrs = Flipped(Vec(nRoCCCSRs, new CustomCSRIO))
}

class RoCCIO(val nPTWPorts: Int, nRoCCCSRs: Int, xLen: Int) extends RoCCCoreIO(nRoCCCSRs, xLen) {
  // val ptw = Vec(nPTWPorts, new TLBPTWIO)
  // val fpu_req = Decoupled(new FPInput)
  // val fpu_resp = Flipped(Decoupled(new FPResult))
}

case class CustomCSR(id: Int, mask: BigInt, init: Option[BigInt])

object CustomCSR {
  def constant(id: Int, value: BigInt): CustomCSR = CustomCSR(id, BigInt(0), Some(value))
}

/** Base classes for Diplomatic TL2 RoCC units **/
//abstract class RoCC(
//  //val opcodes: OpcodeSet,
//  val nPTWPorts: Int = 0,
//  val usesFPU: Boolean = false,
//  val roccCSRs: Seq[CustomCSR] = Nil) extends LazyModule {
//  val module: LazyRoCCModuleImp
//  require(roccCSRs.map(_.id).toSet.size == roccCSRs.size)
//  val atlNode: TLNode = TLIdentityNode()
//  val tlNode: TLNode = TLIdentityNode()
//  val stlNode: TLNode = TLIdentityNode()
//}

//class RoCCModuleImp(outer: LazyRoCC) extends LazyModuleImp(outer) {
//  val io = IO(new RoCCIO(outer.nPTWPorts, outer.roccCSRs.size))
//  io := DontCare
//}
