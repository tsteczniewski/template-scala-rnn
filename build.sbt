assemblySettings

name := "template-scala-rnn"

organization := "io.prediction"

libraryDependencies ++= Seq(
  "io.prediction"      %% "core"           % "0.9.0"   % "provided",
  "org.apache.spark"   %% "spark-core"     % "1.2.0"   % "provided",
  "org.apache.spark"   %% "spark-mllib"    % "1.2.0"   % "provided",
  "org.scalanlp"       %% "breeze"         % "0.11.2",
  "org.scalanlp"       %% "breeze-natives" % "0.11.2",
  "org.apache.opennlp" %  "opennlp-tools"  % "1.5.3",
  "org.scalatest" % "scalatest_2.10" % "latest.integration" % "test"
)
