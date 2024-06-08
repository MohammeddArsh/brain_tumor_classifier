const form = document.getElementById('upload-form');
const fileInput = document.getElementById('file-input');
const resultDiv = document.getElementById('result');
const progressBar = document.getElementById('progress-bar');
const imageContainer = document.getElementById('image-container');

var modal = document.getElementById("popupModal");
var btn = document.getElementById("info");
var span = document.getElementsByClassName("closeBtn")[0];

btn.onclick = function() {
    modal.style.display = "block";
}

span.onclick = function() {
    modal.style.display = "none";
}

window.onclick = function(event) {
    if (event.target == modal) {
        modal.style.display = "none";
    }
}



form.addEventListener('submit', async (event) => {
    event.preventDefault();
    resultDiv.innerHTML = ''; // Clear previous results
    imageContainer.innerHTML = ''; // Clear previous image

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        progressBar.style.display = 'block'; // Show progress bar
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            body: formData,
            // Adding progress event to show upload progress
            onUploadProgress: (progressEvent) => {
                if (progressEvent.lengthComputable) {
                    const percentCompleted = Math.round((progressEvent.loaded / progressEvent.total) * 100);
                    progressBar.value = percentCompleted;
                }
            }
        });
        const data = await response.json();

        if (response.ok) {
            // Display uploaded image
            const imageUrl = URL.createObjectURL(fileInput.files[0]);
            const imageElement = document.createElement('img');
            imageElement.setAttribute('src', imageUrl);
            imageElement.setAttribute('id', 'uploaded-image');
            imageContainer.appendChild(imageElement);

            // Display classification result
            resultDiv.innerHTML = `<p>Predicted Class: ${data.class}</p><p>Confidence: ${data.confidence.toFixed(2)}</p>`;
        } else {
            resultDiv.innerText = `Error: ${data.detail}`;
        }
    } catch (error) {
        console.error('Error:', error);
        resultDiv.innerText = 'An error occurred. Please try again later.';
    } finally {
        progressBar.style.display = 'none'; // Hide progress bar after request completes
    }
});