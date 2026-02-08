import chisel3._


// class PTBR(pgLevels: Int, minPgLevels: Int, xLen: Int, maxPAddrBits: Int, pgIdxBits: Int, usingVM: Bool) extends Bundle {
//   def additionalPgLevels = mode.extract(log2Ceil(pgLevels-minPgLevels+1)-1, 0)
//   def pgLevelsToMode(i: Int) = (xLen, i) match {
//     case (32, 2) => 1
//     case (64, x) if x >= 3 && x <= 6 => x + 5
//   }
//   val (modeBits, maxASIdBits) = xLen match {
//     case 32 => (1, 9)
//     case 64 => (4, 16)
//   }
//   require(!usingVM || modeBits + maxASIdBits + maxPAddrBits - pgIdxBits == xLen)

//   val mode = UInt(modeBits.W)
//   val asid = UInt(maxASIdBits.W)
//   val ppn = UInt((maxPAddrBits - pgIdxBits).W)
// }

object PRV
{
  val SZ = 2
  val U = 0
  val S = 1
  val H = 2
  val M = 3
}

class MStatus extends Bundle {
  // not truly part of mstatus, but convenient
  val debug = Bool()
  val cease = Bool()
  val wfi = Bool()
  val isa = UInt(32.W)

  val dprv = UInt(PRV.SZ.W) // effective prv for data accesses
  val dv = Bool() // effective v for data accesses
  val prv = UInt(PRV.SZ.W)
  val v = Bool()

  val sd = Bool()
  val zero2 = UInt(23.W)
  val mpv = Bool()
  val gva = Bool()
  val mbe = Bool()
  val sbe = Bool()
  val sxl = UInt(2.W)
  val uxl = UInt(2.W)
  val sd_rv32 = Bool()
  val zero1 = UInt(8.W)
  val tsr = Bool()
  val tw = Bool()
  val tvm = Bool()
  val mxr = Bool()
  val sum = Bool()
  val mprv = Bool()
  val xs = UInt(2.W)
  val fs = UInt(2.W)
  val mpp = UInt(2.W)
  val vs = UInt(2.W)
  val spp = UInt(1.W)
  val mpie = Bool()
  val ube = Bool()
  val spie = Bool()
  val upie = Bool()
  val mie = Bool()
  val hie = Bool()
  val sie = Bool()
  val uie = Bool()
}
