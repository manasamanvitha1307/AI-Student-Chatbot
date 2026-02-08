const input = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const chatBody = document.getElementById("chat-body");
const typingIndicator = document.getElementById("typing-indicator");

function addMessage(sender, text) {
    const msg = document.createElement("div");
    msg.className = sender === "user" ? "user-msg" : "bot-msg";
    msg.innerText = text;
    chatBody.appendChild(msg);
    chatBody.scrollTop = chatBody.scrollHeight;
}

function showTyping() {
    typingIndicator.classList.remove("hidden");
}

function hideTyping() {
    typingIndicator.classList.add("hidden");
}

async function sendMessage() {
    const message = input.value.trim();
    if (!message) return;

    addMessage("user", message);
    input.value = "";

    showTyping();

    const response = await fetch("/get", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message })
    });

    const data = await response.json();

    hideTyping();
    addMessage("bot", data.reply);
}

sendBtn.addEventListener("click", sendMessage);
input.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendMessage();
});

/* Auto Welcome Message */
window.onload = () => {
    setTimeout(() => {
        addMessage("bot", "Hi! I'm your ECE Paris AI assistant. Ask me anything about programs, admissions, campus life, or student services.");
    }, 300);
};