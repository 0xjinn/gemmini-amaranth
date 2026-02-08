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
  val busy = Output(Bool())
  val interrupt = Output(Bool())
  val exception = Input(Bool())
}

class RoCCIO(val nPTWPorts: Int, nRoCCCSRs: Int, xLen: Int) extends RoCCCoreIO(nRoCCCSRs, xLen)

case class CustomCSR(id: Int, mask: BigInt, init: Option[BigInt])

object CustomCSR {
  def constant(id: Int, value: BigInt): CustomCSR = CustomCSR(id, BigInt(0), Some(value))
}
