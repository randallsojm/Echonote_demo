const startBtn = document.getElementById("start-btn")
const stopBtn = document.getElementById("stop-btn")
const uploadBtn = document.getElementById("upload-btn")
const audioFileInput = document.getElementById("audioFileInput")
const loadingOverlay = document.getElementById("loading-overlay")
const savedDocumentsHeader = document.getElementById("savedDocumentsHeader")
const documentsList = document.getElementById("documentsList")

let mediaRecorder
let firstSoundTime
let lastSoundTime
let audioChunks = []
let audioBlob = null
let isUploading = false

// Start recording - this will be saved to uploads folder
startBtn.addEventListener("click", async () => {
  if (isUploading) return

  const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  mediaRecorder = new MediaRecorder(stream)

  mediaRecorder.ondataavailable = (event) => {
    audioChunks.push(event.data)
    if (audioChunks.length === 1) {
      firstSoundTime = new Date().toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" })
      console.log("Time of first sound recorded (24-hour format):", firstSoundTime)
    }
  }

  mediaRecorder.onstop = () => {
    audioBlob = new Blob(audioChunks, { type: "audio/wav" })
    audioChunks = []
    // Upload live recording - will be saved to uploads folder
    uploadLiveRecording(audioBlob)
  }

  mediaRecorder.start()

  startBtn.style.display = "none"
  stopBtn.style.display = "inline-block"
  stopBtn.disabled = false
  uploadBtn.disabled = true
})

stopBtn.addEventListener("click", () => {
  mediaRecorder.stop()
  lastSoundTime = new Date().toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" })

  startBtn.style.display = "none"
  uploadBtn.style.display = "none"
  stopBtn.style.display = "none"
  loadingOverlay.style.display = "block"
})

// Upload pre-recorded file - only process, don't save to uploads folder
uploadBtn.addEventListener("click", () => {
  if (isUploading) return
  firstSoundTime = "N.A."
  lastSoundTime = "N.A."
  audioFileInput.click()
})

audioFileInput.addEventListener("change", () => {
  const file = audioFileInput.files[0]
  if (!file || isUploading) {
    console.error("No audio file selected or upload in progress")
    return
  }

  // Process uploaded file - don't save to uploads folder
  processUploadedFile(file)

  startBtn.style.display = "none"
  uploadBtn.style.display = "none"
  stopBtn.style.display = "none"
  loadingOverlay.style.display = "block"
  audioFileInput.value = ""
})

// Function for live recordings - saves to uploads folder
async function uploadLiveRecording(audioBlob) {
  if (isUploading) return
  isUploading = true

  const formData = new FormData()
  formData.append("audio", audioBlob, "live-recording.wav")
  formData.append("firstSoundTime", firstSoundTime)
  formData.append("lastSoundTime", lastSoundTime)
  formData.append("recordingType", "live") // Flag to indicate this should be saved

  try {
    const response = await fetch("http://localhost:3001/upload-live", {
      method: "POST",
      body: formData,
    })

    const result = await response.json()
    console.log("Server response:", result)

    if (response.ok) {
      alert("Live recording uploaded and processed successfully!")
      fetchSavedDocuments()
    } else {
      alert(`Failed to process audio: ${JSON.stringify(result)}`)
    }
  } catch (err) {
    console.error("Error uploading live recording:", err)
    alert(`Upload error: ${err.message}`)
  } finally {
    resetUI()
  }
}

// Function for uploaded files - only processes, doesn't save
async function processUploadedFile(file) {
  if (isUploading) return
  isUploading = true

  const formData = new FormData()
  formData.append("audio", file)
  formData.append("fileName", file.name)
  formData.append("filePath", file.webkitRelativePath || file.name) // Get relative path if available
  formData.append("recordingType", "uploaded") // Flag to indicate this shouldn't be saved

  try {
    const response = await fetch("http://localhost:3001/process-uploaded", {
      method: "POST",
      body: formData,
    })

    const result = await response.json()
    console.log("Server response:", result)

    if (response.ok) {
      alert("Uploaded file processed successfully!")
      fetchSavedDocuments()
    } else {
      alert(`Failed to process audio: ${JSON.stringify(result)}`)
    }
  } catch (err) {
    console.error("Error processing uploaded file:", err)
    alert(`Processing error: ${err.message}`)
  } finally {
    resetUI()
  }
}

// Helper function to reset UI
function resetUI() {
  loadingOverlay.style.display = "none"
  startBtn.style.display = "inline-block"
  uploadBtn.style.display = "inline-block"
  stopBtn.style.display = "none"
  isUploading = false
  uploadBtn.disabled = false
}

// Function to fetch and display saved documents
async function fetchSavedDocuments() {
  try {
    const response = await fetch("http://localhost:3001/documents")
    const data = await response.json()

    console.log("Fetched documents:", data)

    if (data.documents.length === 0) {
      savedDocumentsHeader.style.display = "none"
      documentsList.innerHTML = "<p>No documents found.</p>"
    } else {
      savedDocumentsHeader.style.display = "block"
      documentsList.innerHTML = ""

      data.documents.forEach((doc) => {
        const docElement = document.createElement("div")
        docElement.classList.add("document-card")

        docElement.innerHTML = `
                    <h3>${doc.title}</h3>
                    <a href="http://localhost:3001/documents/${doc.filename}" target="_blank">View Document</a>
                `

        documentsList.appendChild(docElement)
      })
    }
  } catch (error) {
    console.error("Error fetching documents:", error)
  }
}
