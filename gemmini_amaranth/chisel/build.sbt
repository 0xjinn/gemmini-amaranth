val chisel3Version = "3.6.1"

lazy val chisel3Settings = Seq(
  libraryDependencies ++= Seq("edu.berkeley.cs" %% "chisel3" % chisel3Version),
  addCompilerPlugin("edu.berkeley.cs" % "chisel3-plugin" % chisel3Version cross CrossVersion.full)
)

lazy val chiselSettings = chisel3Settings ++ Seq(
  libraryDependencies ++= Seq(
    "org.apache.commons" % "commons-lang3" % "3.12.0",
    "org.apache.commons" % "commons-text" % "1.9"
  )
)

lazy val commonSettings = Seq(
  organization := "edu.berkeley.cs",
  version := "1.6",
  scalaVersion := "2.13.12",
  libraryDependencies += "com.lihaoyi" %% "sourcecode" % "0.3.1",
  exportJars := true,
  resolvers ++= Seq(
    Resolver.sonatypeRepo("snapshots"),
    Resolver.sonatypeRepo("releases"),
    Resolver.mavenLocal),
  allDependencies := {
    val dropDeps = Seq(("edu.berkeley.cs", "rocketchip"))
    allDependencies.value.filterNot { dep =>
      dropDeps.contains((dep.organization, dep.name))
    }
  })

lazy val root = Project("gemmini-amaranth", file("."))
  .dependsOn(hardfloat)
  .settings(commonSettings)
  .settings(chiselSettings)

def freshProject(name: String, dir: File): Project = {
  Project(id = name, base = dir / "src")
    .settings(
      Compile / scalaSource := baseDirectory.value / "main" / "scala",
      Compile / resourceDirectory := baseDirectory.value / "main" / "resources"
    )
}

lazy val cde = (project in file("cde"))
  .settings(commonSettings)
  .settings(Compile / scalaSource := baseDirectory.value / "cde/src/chipsalliance/rocketchip")

lazy val hardfloat = freshProject("berkeley-hardfloat", file("berkeley-hardfloat/hardfloat"))
  .settings(chiselSettings)
  .settings(commonSettings)
