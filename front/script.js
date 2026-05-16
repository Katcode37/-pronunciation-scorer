let mediaRecorder;
let audioChunks = [];
let recordedBlob = null;

const textSelect = document.getElementById("textSelect");
const textToRead = document.getElementById("textToRead");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const sendBtn = document.getElementById("sendBtn");

const statusText = document.getElementById("status");
const resultDiv = document.getElementById("result");

textSelect.addEventListener("change", () => {
  textToRead.textContent = textSelect.value;
});

startBtn.addEventListener("click", async () => {
  audioChunks = [];
  recordedBlob = null;

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: true
  });

  mediaRecorder = new MediaRecorder(stream);

  mediaRecorder.ondataavailable = (event) => {
    audioChunks.push(event.data);
  };

  mediaRecorder.onstop = () => {
    recordedBlob = new Blob(audioChunks, {
      type: "audio/webm"
    });

    sendBtn.disabled = false;
    statusText.textContent = "Recording finished. Ready to send.";
  };

  mediaRecorder.start();

  startBtn.disabled = true;
  stopBtn.disabled = false;
  sendBtn.disabled = true;

  statusText.textContent = "Recording...";
});

stopBtn.addEventListener("click", () => {
  mediaRecorder.stop();

  startBtn.disabled = false;
  stopBtn.disabled = true;
});

sendBtn.addEventListener("click", async () => {
  const formData = new FormData();

  formData.append("audio", recordedBlob, "recording.webm");
  formData.append("expected_text", textSelect.value);

  statusText.textContent = "Sending audio...";

  resultDiv.innerHTML = `
    <div class="loading-box">
      <div class="spinner"></div>
      <p id="loadingMessage"></p>
    </div>
  `;

  const loadingMessage = document.getElementById("loadingMessage");

  const loadingTexts = [
    "Kate worked hard on it, let's forgive her a bit longer waiting time.",
    "It is hard to process speech, please be a patient cookie.",
    "Analyzing pronunciation magic..."
  ];

  let currentMessageIndex = 0;
  let loadingInterval;
  let loadingTimeout;

  loadingTimeout = setTimeout(() => {
    statusText.textContent = "";
    loadingMessage.textContent = loadingTexts[0];

    loadingInterval = setInterval(() => {
      currentMessageIndex =
        (currentMessageIndex + 1) % loadingTexts.length;

      loadingMessage.textContent =
        loadingTexts[currentMessageIndex];
    }, 6000);
  }, 5000);

  const response = await fetch("https://hunter-remover-struck.ngrok-free.dev/score", {
    method: "POST",
    body: formData
  });

  const data = await response.json();

  clearTimeout(loadingTimeout);
  clearInterval(loadingInterval);

  resultDiv.innerHTML = `
    <p><strong>Score:</strong> ${data.score}</p>
    <p><strong>Feedback:</strong> ${data.feedback}</p>

    ${
      data.text_match_score < 70
        ? `<p><strong>Text match:</strong> ${data.text_match_score}%</p>`
        : ""
    }

    ${
      data.weakest_words && data.weakest_words.length > 0
        ? `<p><strong>Weakest words:</strong> ${
            data.weakest_words.map(item => item.word).join(", ")
          }</p>`
        : `<p><strong>All words are more or less OK.</strong></p>`
    }
  `;

  statusText.textContent = "Done.";
});
