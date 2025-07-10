const express = require("express")
const multer = require("multer")
const path = require("path")
const { exec, spawn } = require("child_process")
const fs = require("fs")
const cors = require("cors")

const app = express()
const port = 3001

app.use(cors())
app.use((req, res, next) => {
  console.log(`Received ${req.method} request to ${req.url}`)
  next()
})

// Ensure directories exist
const uploadDir = path.join(__dirname, "uploads")
const processedDir = path.join(__dirname, "processed")
const tempDir = path.join(__dirname, "temp")
;[uploadDir, processedDir, tempDir].forEach((dir) => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true })
  }
})

// Multer configurations
const liveStorage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => cb(null, Date.now() + path.extname(file.originalname)),
})

const uploadedStorage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, tempDir),
  filename: (req, file, cb) => cb(null, "temp_" + Date.now() + path.extname(file.originalname)),
})

const uploadLive = multer({ storage: liveStorage })
const uploadFile = multer({ storage: uploadedStorage })

// Serve static files
app.use("/uploads", express.static(uploadDir))
app.use("/processed", express.static(processedDir))

// COMPREHENSIVE PIPELINE PROCESSING
function processAudioWithPython(audioFilePath, firstTime, lastTime, res, shouldCleanup = false) {
  const first = firstTime || "N.A."
  const last = lastTime || "N.A."

  console.log(`=== STARTING COMPREHENSIVE AUDIO PROCESSING ===`)
  console.log(`Audio file: ${audioFilePath}`)
  console.log(`Meeting time: ${first} - ${last}`)
  console.log(`Working directory: ${__dirname}`)

  const normalizedPath = audioFilePath.replace(/\\/g, "/")
  const isWindows = process.platform === "win32"

  let command, args

  if (isWindows) {
    command = "cmd"
    args = [
      "/c",
      `cd /d "${__dirname}" && conda activate diarisation && python main_pipeline.py "${normalizedPath}" "${first}" "${last}"`,
    ]
  } else {
    command = "bash"
    args = ["-c", `cd "${__dirname}" && python main_pipeline.py "${normalizedPath}" "${first}" "${last}"`]
  }

  console.log(`Executing comprehensive pipeline: ${command} ${args.join(" ")}`)
  console.log("Pipeline includes:")
  console.log("  1. Speaker diarisation (preserving A, B, C labels)")
  console.log("  2. High-quality topic extraction with semantic analysis")
  console.log("  3. RAG-based point extraction using FAISS")
  console.log("  4. Comprehensive speaker analysis (participation, sentiment, style)")
  console.log("  5. Professional document generation")
  console.log("  6. Podcast script with speaker insights")
  console.log("  7. High-quality TTS audio generation")

  const pythonProcess = spawn(command, args, {
    cwd: __dirname,
    stdio: ["pipe", "pipe", "pipe"],
    shell: true,
    env: {
      ...process.env,
      PYTHONPATH: __dirname,
      PYTHONUNBUFFERED: "1",
    },
  })

  let stdout = ""
  let stderr = ""
  let lastProgressTime = Date.now()

  pythonProcess.stdout.on("data", (data) => {
    const output = data.toString()
    console.log("Pipeline output:", output)
    stdout += output
    lastProgressTime = Date.now()
  })

  pythonProcess.stderr.on("data", (data) => {
    const error = data.toString()
    if (!error.includes("UserWarning") && !error.includes("LangChainDeprecationWarning")) {
      console.log("Pipeline stderr:", error)
    }
    stderr += error
    lastProgressTime = Date.now()
  })

  pythonProcess.on("close", (code) => {
    console.log(`Comprehensive pipeline exited with code: ${code}`)

    if (shouldCleanup && fs.existsSync(audioFilePath)) {
      fs.unlink(audioFilePath, (unlinkErr) => {
        if (unlinkErr) console.error("Error deleting temp file:", unlinkErr)
        else console.log("Temp file deleted:", audioFilePath)
      })
    }

    if (code !== 0) {
      console.error("Comprehensive pipeline failed with code:", code)
      console.error("Full stderr:", stderr)
      return res.status(500).json({
        error: "Comprehensive audio processing failed",
        details: stderr || `Process exited with code ${code}`,
        stdout: stdout,
        exitCode: code,
      })
    }

    console.log("=== COMPREHENSIVE PIPELINE COMPLETED SUCCESSFULLY ===")
    console.log("Generated high-quality analysis files:")
    console.log("  - Comprehensive_Analysis_Report.txt (detailed analysis)")
    console.log("  - Podcast_Script.txt (professional script)")
    console.log("  - Meeting_Analysis_Audio.wav (audio summary)")

    res.json({
      message: "Comprehensive audio analysis completed successfully",
      filePath: audioFilePath,
      output: stdout,
      analysis_features: [
        "High-quality topic extraction with semantic analysis",
        "Comprehensive speaker analysis (participation, sentiment, communication style)",
        "RAG-based point extraction using FAISS indexing",
        "Professional document generation with detailed insights",
        "Podcast script with speaker-specific analysis",
        "High-quality TTS audio generation",
      ],
      generated_files: [
        "Comprehensive_Analysis_Report.txt",
        "Podcast_Script.txt",
        "Meeting_Analysis_Audio.wav",
      ],
    })
  })

  pythonProcess.on("error", (err) => {
    console.error("Failed to start comprehensive pipeline:", err)

    if (shouldCleanup && fs.existsSync(audioFilePath)) {
      fs.unlink(audioFilePath, () => {})
    }

    res.status(500).json({
      error: "Failed to start comprehensive audio processing",
      details: err.message,
    })
  })

  // Extended timeout for comprehensive analysis (25 minutes)
  const TIMEOUT_DURATION = 25 * 60 * 1000

  const timeout = setTimeout(() => {
    console.log("Comprehensive pipeline timeout - killing process")
    pythonProcess.kill("SIGTERM")

    if (shouldCleanup && fs.existsSync(audioFilePath)) {
      fs.unlink(audioFilePath, () => {})
    }

    if (!res.headersSent) {
      res.status(500).json({
        error: "Comprehensive processing timeout",
        details: `Process took longer than ${TIMEOUT_DURATION / 60000} minutes to complete`,
      })
    }
  }, TIMEOUT_DURATION)

  // Progress monitoring every 3 minutes
  const progressCheck = setInterval(
    () => {
      const timeSinceLastProgress = Date.now() - lastProgressTime
      const maxIdleTime = 8 * 60 * 1000 // 8 minutes of no output

      if (timeSinceLastProgress > maxIdleTime) {
        console.log("No progress detected for 8 minutes - killing process")
        clearInterval(progressCheck)
        pythonProcess.kill("SIGTERM")

        if (!res.headersSent) {
          res.status(500).json({
            error: "Process appears stuck",
            details: "No output received for 8 minutes",
          })
        }
      } else {
        console.log(`Comprehensive pipeline active - last output ${Math.round(timeSinceLastProgress / 1000)}s ago`)
      }
    },
    3 * 60 * 1000,
  )

  pythonProcess.on("close", () => {
    clearTimeout(timeout)
    clearInterval(progressCheck)
  })
}

// Upload endpoints
app.post("/upload-live", uploadLive.single("audio"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No audio file received" })
  }

  const audioFilePath = req.file.path
  const firstTime = req.body.firstSoundTime
  const lastTime = req.body.lastSoundTime

  console.log(`Live recording saved at: ${audioFilePath}`)
  processAudioWithPython(audioFilePath, firstTime, lastTime, res, false)
})

app.post("/process-uploaded", uploadFile.single("audio"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No audio file received" })
  }

  const audioFilePath = req.file.path
  console.log(`Uploaded file temporarily saved at: ${audioFilePath}`)
  processAudioWithPython(audioFilePath, "N.A.", "N.A.", res, true)
})

// Debug endpoints
app.get("/test-python", (req, res) => {
  const isWindows = process.platform === "win32"
  const command = isWindows
    ? `cmd /c "conda activate diarisation && cd /d "${__dirname}" && python --version && python -c "import sys; print('Python path:', sys.path)" && dir"`
    : `bash -c "cd '${__dirname}' && python --version && python -c 'import sys; print(\"Python path:\", sys.path)' && ls -la"`

  exec(command, { cwd: __dirname, shell: true }, (err, stdout, stderr) => {
    res.json({
      error: err ? err.message : null,
      stdout: stdout,
      stderr: stderr,
      cwd: __dirname,
      platform: process.platform,
    })
  })
})

app.get("/debug-files", (req, res) => {
  const files = {
    mainPipeline: fs.existsSync(path.join(__dirname, "main_pipeline.py")),
    speakerDiarisation: fs.existsSync(path.join(__dirname, "speaker_diarisation.py")),
    mainLLMWorkflow: fs.existsSync(path.join(__dirname, "main_llm_workflow.py")),
    topicExtraction: fs.existsSync(path.join(__dirname, "topic_extraction.py")),
    ragPointExtraction: fs.existsSync(path.join(__dirname, "rag_point_extraction.py")),
    speakerAnalysis: fs.existsSync(path.join(__dirname, "speaker_analysis.py")),
    documentFormatter: fs.existsSync(path.join(__dirname, "document_formatter.py")),
    podcastGenerator: fs.existsSync(path.join(__dirname, "podcast_generator.py")),
    ttsPhase: fs.existsSync(path.join(__dirname, "TTS_phase_3_offline.py")),
    configYaml: fs.existsSync(path.join(__dirname, "config.yaml")),
    utils: fs.existsSync(path.join(__dirname, "utils.py")),
    uploadsDir: fs.existsSync(uploadDir),
    processedDir: fs.existsSync(processedDir),
  }

  res.json({
    workingDirectory: __dirname,
    files: files,
    allFilesExist: Object.values(files).every((exists) => exists),
    pipelineType: "Comprehensive high-quality analysis pipeline",
    features: [
      "Speaker diarisation with original labels (A, B, C)",
      "High-quality topic extraction with semantic analysis",
      "RAG-based point extraction using FAISS",
      "Comprehensive speaker analysis (participation, sentiment, style)",
      "Professional document generation",
      "Podcast script with speaker insights",
      "High-quality TTS audio generation",
    ],
    expectedOutputFiles: [
      "Comprehensive_Analysis_Report.txt",
      "Podcast_Script.txt",
      "Meeting_Analysis_Audio.wav",
    ],
  })
})

// Document serving
app.get("/documents", (req, res) => {
  fs.readdir(processedDir, (err, files) => {
    if (err) {
      return res.status(500).json({ error: "Unable to retrieve documents" })
    }

    const documents = files
      .filter((file) => !file.startsWith("~lock") && !file.endsWith("#"))
      .map((file) => ({
        title: file,
        filename: file,
      }))

    res.json({ documents })
  })
})

app.get("/documents/:filename", (req, res) => {
  const filePath = path.join(processedDir, req.params.filename)
  if (!fs.existsSync(filePath)) {
    return res.status(404).json({ error: "File not found" })
  }
  res.sendFile(filePath)
})

// Cleanup temp directory
function cleanupTempDirectory() {
  fs.readdir(tempDir, (err, files) => {
    if (err) return
    files.forEach((file) => {
      const filePath = path.join(tempDir, file)
      fs.unlink(filePath, () => {})
    })
  })
}

cleanupTempDirectory()

app
  .listen(port, () => {
    console.log(`=== COMPREHENSIVE MEETING ANALYSIS SERVER ===`)
    console.log(`Server running on: http://localhost:${port}`)
    console.log(`Upload directory: ${uploadDir}`)
    console.log(`Processed directory: ${processedDir}`)
    console.log(`Platform: ${process.platform}`)
    console.log("")
    console.log("=== COMPREHENSIVE PIPELINE FEATURES ===")
    console.log("✓ Speaker diarisation (preserves original A, B, C labels)")
    console.log("✓ High-quality topic extraction with semantic analysis")
    console.log("✓ RAG-based point extraction using FAISS indexing")
    console.log("✓ Comprehensive speaker analysis:")
    console.log("  - Participation metrics (speaking time, word count, segments)")
    console.log("  - Sentiment analysis (positive/negative/neutral)")
    console.log("  - Communication style analysis (formality, complexity)")
    console.log("  - Interaction patterns (agreements/disagreements)")
    console.log("  - Comparative analysis between speakers")
    console.log("✓ Professional document generation with detailed insights")
    console.log("✓ Podcast script with speaker-specific analysis")
    console.log("✓ High-quality TTS audio generation")
    console.log("")
    console.log("Expected output files:")
    console.log("  - Comprehensive_Analysis_Report.txt (detailed analysis)")
    console.log("  - Podcast_Script.txt (professional script)")
    console.log("  - Meeting_Analysis_Audio.wav (audio summary)")
    console.log("")
    console.log("Debug endpoints:")
    console.log("  - Test Python: http://localhost:3001/test-python")
    console.log("  - Debug Files: http://localhost:3001/debug-files")
    console.log("")
    console.log("TIMEOUT SETTINGS:")
    console.log("  - Main timeout: 25 minutes")
    console.log("  - Idle timeout: 8 minutes")
    console.log("  - Progress check: every 3 minutes")
  })
  .setTimeout(0)
