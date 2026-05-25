const API_URL = "https://vitoriia-animal-weight-api.hf.space/predict";

// =========================
// ELEMENTS
// =========================
const fileInput = document.getElementById("fileInput");
const uploadImageBtn = document.getElementById("uploadImageBtn");
const uploadBtn = document.getElementById("uploadBtn");

const preview = document.getElementById("preview");
const placeholder = document.getElementById("placeholder");

const resultContent = document.getElementById("resultContent");

// =========================
// STATE
// =========================
let selected = null;

// =========================
// UI
// =========================
function showPreview(src) {
    preview.src = src;
    preview.style.display = "block";
    placeholder.style.display = "none";
}

function resetSelection() {
    selected = null;
    preview.src = "";
    preview.style.display = "none";
    placeholder.style.display = "block";
}

// =========================
// UPLOAD IMAGE
// =========================
uploadImageBtn.addEventListener("click", () => {
    fileInput.click();
});

fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;

    selected = {
        file: file,
        name: file.name
    };

    showPreview(URL.createObjectURL(file));
});

// =========================
// PREDICT
// =========================
uploadBtn.addEventListener("click", async () => {

    if (!selected) {
        alert("Оберіть фото");
        return;
    }

    sendImage(selected);
});

// =========================
// SEND IMAGE
// =========================
async function sendImage(item) {

    resultContent.innerHTML = `<p>Обробка...</p>`;

    const formData = new FormData();

    try {

        formData.append("file", item.file, item.name);

        const response = await fetch(API_URL, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!data.success) {
            resultContent.innerHTML = `<p class="error">${data.error}</p>`;
            return;
        }

        if (!data.is_cow) {
            resultContent.innerHTML = `<p class="error">Корову не знайдено</p>`;
            return;
        }

        resultContent.innerHTML = `
            <p class="success">Корова визначена</p>

            <div class="result-item">
                <div class="result-label">Вага</div>
                <div class="result-value">${data.predicted_weight} кг</div>
            </div>

            <div class="result-item">
                <div class="result-label">MAPE</div>
                <div class="result-value">${data.mape}%</div>
            </div>

            <div class="result-item">
                <div class="result-label">Accuracy</div>
                <div class="result-value">${data.classifier_confidence}%</div>
            </div>
        `;

    } catch (err) {
        console.error(err);
        resultContent.innerHTML = `<p class="error">Помилка з'єднання з сервером</p>`;
    }
}