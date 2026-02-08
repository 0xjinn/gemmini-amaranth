import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import chisel3._
import hardfloat._

import Arithmetic.SIntArithmetic
import Arithmetic.FloatArithmetic._

object GemminiGenerator {

  // ---------------------------------------------------------------------------
  // Argument parsing helpers
  // ---------------------------------------------------------------------------
  private def parseGemminiArgs(args: Array[String]): Map[String, String] = {
    args.filter(_.startsWith("--gemmini-")).map { arg =>
      val stripped = arg.stripPrefix("--gemmini-")
      val idx = stripped.indexOf('=')
      if (idx < 0) (stripped, "true")
      else (stripped.substring(0, idx), stripped.substring(idx + 1))
    }.toMap
  }

  private def chiselArgs(args: Array[String]): Array[String] =
    args.filterNot(_.startsWith("--gemmini-"))

  // ---------------------------------------------------------------------------
  // SInt scale lambdas (reused from Configs.scala defaultConfig)
  // ---------------------------------------------------------------------------
  private def sintMvinScale(inputWidth: Int): ScaleArguments[SInt, Float] = {
    ScaleArguments(
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
        val maxsat = ((1 << (t.getWidth - 1)) - 1).S
        val minsat = (-(1 << (t.getWidth - 1))).S
        val sign = rawFloatFromRecFN(f.expWidth, f.sigWidth, rec_fn_to_in.io.in).sign
        val sat = Mux(sign, minsat, maxsat)

        Mux(overflow, sat, rec_fn_to_in.io.out.asTypeOf(t))
      },
      4, Float(8, 24), 4,
      identity = "1.0",
      c_str = s"({float y = ROUND_NEAR_EVEN((x) * (scale)); y > INT${inputWidth / 2 * 2}_MAX ? INT${inputWidth / 2 * 2}_MAX : (y < INT${inputWidth / 2 * 2}_MIN ? INT${inputWidth / 2 * 2}_MIN : (elem_t)y);})"
    )
  }

  private def sintAccScale(accWidth: Int): ScaleArguments[SInt, Float] = {
    ScaleArguments(
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
        val maxsat = ((1 << (t.getWidth - 1)) - 1).S
        val minsat = (-(1 << (t.getWidth - 1))).S
        val sign = rawFloatFromRecFN(f.expWidth, f.sigWidth, rec_fn_to_in.io.in).sign
        val sat = Mux(sign, minsat, maxsat)

        Mux(overflow, sat, rec_fn_to_in.io.out.asTypeOf(t))
      },
      8, Float(8, 24), -1,
      identity = "1.0",
      c_str = s"({float y = ROUND_NEAR_EVEN((x) * (scale)); y > INT${accWidth / 2 * 2}_MAX ? INT${accWidth / 2 * 2}_MAX : (y < INT${accWidth / 2 * 2}_MIN ? INT${accWidth / 2 * 2}_MIN : (acc_t)y);})"
    )
  }

  // ---------------------------------------------------------------------------
  // Build configs from parsed args
  // ---------------------------------------------------------------------------
  private def getInt(m: Map[String, String], key: String, default: Int): Int =
    m.getOrElse(key, default.toString).toInt

  private def getBool(m: Map[String, String], key: String, default: Boolean): Boolean =
    m.getOrElse(key, default.toString).toBoolean

  private def getDataflow(m: Map[String, String]): Dataflow.Value =
    m.getOrElse("dataflow", "BOTH") match {
      case "OS"   => Dataflow.OS
      case "WS"   => Dataflow.WS
      case "BOTH" => Dataflow.BOTH
      case other  => throw new IllegalArgumentException(s"Unknown dataflow: $other")
    }

  private def buildSIntConfig(m: Map[String, String]): GemminiArrayConfig[SInt, Float, Float] = {
    val inputWidth = getInt(m, "inputWidth", 8)
    val accWidth = getInt(m, "accWidth", 32)
    val spatialWidth = getInt(m, "spatialOutputWidth", 20)

    GemminiConfigs.defaultConfig.copy(
      inputType = SInt(inputWidth.W),
      accType = SInt(accWidth.W),
      spatialArrayOutputType = SInt(spatialWidth.W),
      meshRows = getInt(m, "meshRows", 16),
      meshColumns = getInt(m, "meshColumns", 16),
      tileRows = getInt(m, "tileRows", 1),
      tileColumns = getInt(m, "tileColumns", 1),
      dataflow = getDataflow(m),
      sp_capacity = CapacityInKilobytes(getInt(m, "spCapacityKB", 256)),
      acc_capacity = CapacityInKilobytes(getInt(m, "accCapacityKB", 64)),
      sp_banks = getInt(m, "spBanks", 4),
      acc_banks = getInt(m, "accBanks", 2),
      dma_maxbytes = getInt(m, "dmaMaxbytes", 64),
      dma_buswidth = getInt(m, "dmaBuswidth", 128),
      max_in_flight_mem_reqs = getInt(m, "maxInFlightMemReqs", 16),
      has_training_convs = getBool(m, "hasTrainingConvs", true),
      has_max_pool = getBool(m, "hasMaxPool", true),
      has_nonlinear_activations = getBool(m, "hasNonlinearActivations", true),
      mvin_scale_args = Some(sintMvinScale(inputWidth)),
      acc_scale_args = Some(sintAccScale(accWidth)),
      headerFileName = m.getOrElse("headerFileName", "gemmini_params.h")
    )
  }

  private def buildFloatConfig(m: Map[String, String]): GemminiArrayConfig[Float, Float, Float] = {
    val inputExpWidth = getInt(m, "inputExpWidth", 8)
    val inputSigWidth = getInt(m, "inputSigWidth", 24)
    val accExpWidth = getInt(m, "accExpWidth", 8)
    val accSigWidth = getInt(m, "accSigWidth", 24)

    val inputType = Float(inputExpWidth, inputSigWidth)
    val accType = Float(accExpWidth, accSigWidth)

    GemminiFPConfigs.defaultFPConfig.copy(
      inputType = inputType,
      accType = accType,
      spatialArrayOutputType = inputType,
      meshRows = getInt(m, "meshRows", 4),
      meshColumns = getInt(m, "meshColumns", 4),
      tileRows = getInt(m, "tileRows", 1),
      tileColumns = getInt(m, "tileColumns", 1),
      dataflow = getDataflow(m),
      sp_capacity = CapacityInKilobytes(getInt(m, "spCapacityKB", 256)),
      acc_capacity = CapacityInKilobytes(getInt(m, "accCapacityKB", 64)),
      sp_banks = getInt(m, "spBanks", 4),
      acc_banks = getInt(m, "accBanks", 1),
      dma_maxbytes = getInt(m, "dmaMaxbytes", 64),
      dma_buswidth = getInt(m, "dmaBuswidth", 128),
      max_in_flight_mem_reqs = getInt(m, "maxInFlightMemReqs", 16),
      has_training_convs = getBool(m, "hasTrainingConvs", false),
      has_max_pool = getBool(m, "hasMaxPool", true),
      has_nonlinear_activations = getBool(m, "hasNonlinearActivations", true),
      tile_latency = getInt(m, "tileLatency", 2),
      mvin_scale_args = Some(ScaleArguments((t: Float, u: Float) => t * u, 4, inputType, -1, identity = "1.0", c_str = "((x) * (scale))")),
      mvin_scale_acc_args = Some(ScaleArguments((t: Float, u: Float) => t * u, 4, inputType, -1, identity = "1.0", c_str = "((x) * (scale))")),
      acc_scale_args = Some(ScaleArguments((t: Float, u: Float) => t * u, 4, Float(8, 24), -1, identity = "1.0", c_str = "((x) * (scale))")),
      headerFileName = m.getOrElse("headerFileName", "gemmini_params.h")
    )
  }

  private def buildDummyConfig(m: Map[String, String]): GemminiArrayConfig[DummySInt, Float, Float] = {
    val inputWidth = getInt(m, "inputWidth", 8)
    val accWidth = getInt(m, "accWidth", 32)
    val spatialWidth = getInt(m, "spatialOutputWidth", 20)

    GemminiConfigs.dummyConfig.copy(
      inputType = DummySInt(inputWidth),
      accType = DummySInt(accWidth),
      spatialArrayOutputType = DummySInt(spatialWidth),
      meshRows = getInt(m, "meshRows", 16),
      meshColumns = getInt(m, "meshColumns", 16),
      tileRows = getInt(m, "tileRows", 1),
      tileColumns = getInt(m, "tileColumns", 1),
      dataflow = getDataflow(m),
      sp_capacity = CapacityInKilobytes(getInt(m, "spCapacityKB", 128)),
      acc_capacity = CapacityInKilobytes(getInt(m, "accCapacityKB", 128)),
      sp_banks = getInt(m, "spBanks", 4),
      acc_banks = getInt(m, "accBanks", 2),
      dma_maxbytes = getInt(m, "dmaMaxbytes", 64),
      dma_buswidth = getInt(m, "dmaBuswidth", 128),
      max_in_flight_mem_reqs = getInt(m, "maxInFlightMemReqs", 16),
      has_training_convs = getBool(m, "hasTrainingConvs", false),
      has_max_pool = getBool(m, "hasMaxPool", true),
      has_nonlinear_activations = getBool(m, "hasNonlinearActivations", false),
      headerFileName = m.getOrElse("headerFileName", "gemmini_params.h")
    )
  }

  // ---------------------------------------------------------------------------
  // Preset resolution
  // ---------------------------------------------------------------------------
  private def applyPreset(preset: String, m: Map[String, String]): Map[String, String] = {
    val presetDefaults: Map[String, String] = preset match {
      case "default" => Map()
      case "chip" => Map(
        "inputTypeFamily" -> "sint", "inputWidth" -> "8", "accWidth" -> "32",
        "spCapacityKB" -> "64", "accCapacityKB" -> "32", "dataflow" -> "WS"
      )
      case "largeChip" => Map(
        "inputTypeFamily" -> "sint", "inputWidth" -> "8", "accWidth" -> "32",
        "spCapacityKB" -> "128", "accCapacityKB" -> "64", "dataflow" -> "WS",
        "meshRows" -> "32", "meshColumns" -> "32"
      )
      case "lean" => Map(
        "inputTypeFamily" -> "sint", "inputWidth" -> "8", "accWidth" -> "32",
        "dataflow" -> "WS", "maxInFlightMemReqs" -> "64"
      )
      case "fp32" => Map(
        "inputTypeFamily" -> "float",
        "inputExpWidth" -> "8", "inputSigWidth" -> "24",
        "accExpWidth" -> "8", "accSigWidth" -> "24",
        "meshRows" -> "4", "meshColumns" -> "4"
      )
      case "fp16" => Map(
        "inputTypeFamily" -> "float",
        "inputExpWidth" -> "5", "inputSigWidth" -> "11",
        "accExpWidth" -> "8", "accSigWidth" -> "24",
        "meshRows" -> "4", "meshColumns" -> "4"
      )
      case "bf16" => Map(
        "inputTypeFamily" -> "float",
        "inputExpWidth" -> "8", "inputSigWidth" -> "8",
        "accExpWidth" -> "8", "accSigWidth" -> "24",
        "meshRows" -> "4", "meshColumns" -> "4"
      )
      case other => throw new IllegalArgumentException(s"Unknown preset: $other")
    }
    // Explicit CLI args override preset defaults
    presetDefaults ++ m
  }

  // ---------------------------------------------------------------------------
  // Main
  // ---------------------------------------------------------------------------
  def main(args: Array[String]): Unit = {
    var gemminiParams = parseGemminiArgs(args)
    val cArgs = chiselArgs(args)

    // Apply preset if specified
    gemminiParams.get("preset").foreach { preset =>
      gemminiParams = applyPreset(preset, gemminiParams - "preset")
    }

    val family = gemminiParams.getOrElse("inputTypeFamily", "sint")
    val targetDir = cArgs.sliding(2).collectFirst {
      case Array("--target-dir", dir) => dir
    }.getOrElse(".")

    println(s"[GemminiGenerator] Type family: $family")
    println(s"[GemminiGenerator] Target dir: $targetDir")

    family match {
      case "sint" =>
        val config = buildSIntConfig(gemminiParams)
        println(s"[GemminiGenerator] SInt config: ${config.meshRows}x${config.meshColumns} mesh, " +
          s"${config.inputType.getWidth}-bit input, ${config.accType.getWidth}-bit acc, " +
          s"dataflow=${config.dataflow}")
        emitVerilog(new Gemmini(config), cArgs)
        writeHeader(config.generateHeader(), targetDir, config.headerFileName)
        writeJson(config.generateJson(), targetDir)

      case "float" =>
        val config = buildFloatConfig(gemminiParams)
        println(s"[GemminiGenerator] Float config: ${config.meshRows}x${config.meshColumns} mesh, " +
          s"Float(${config.inputType.expWidth},${config.inputType.sigWidth}) input, " +
          s"Float(${config.accType.expWidth},${config.accType.sigWidth}) acc, " +
          s"dataflow=${config.dataflow}")
        emitVerilog(new Gemmini(config), cArgs)
        writeHeader(config.generateHeader(), targetDir, config.headerFileName)
        writeJson(config.generateJson(), targetDir)

      case "dummy" =>
        val config = buildDummyConfig(gemminiParams)
        println(s"[GemminiGenerator] Dummy config: ${config.meshRows}x${config.meshColumns} mesh")
        emitVerilog(new Gemmini(config), cArgs)
        writeHeader(config.generateHeader(), targetDir, config.headerFileName)
        writeJson(config.generateJson(), targetDir)

      case other =>
        throw new IllegalArgumentException(s"Unknown inputTypeFamily: $other. Use sint, float, or dummy.")
    }

    println("[GemminiGenerator] Done.")
  }

  private def writeHeader(content: String, targetDir: String, fileName: String): Unit = {
    val dir = Paths.get(targetDir)
    Files.createDirectories(dir)
    val path = dir.resolve(fileName)
    Files.write(path, content.getBytes(StandardCharsets.UTF_8))
    println(s"[GemminiGenerator] Wrote header to $path")
  }

  private def writeJson(content: String, targetDir: String): Unit = {
    val dir = Paths.get(targetDir)
    Files.createDirectories(dir)
    val path = dir.resolve("gemmini_config.json")
    Files.write(path, content.getBytes(StandardCharsets.UTF_8))
    println(s"[GemminiGenerator] Wrote config JSON to $path")
  }
}
