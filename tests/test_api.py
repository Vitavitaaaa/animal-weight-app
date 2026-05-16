import io


def test_home(client):

    response = client.get("/")

    assert response.status_code == 200
    assert response.json["message"] == "Cow Weight Prediction API is running"


def test_predict_no_file(client):

    response = client.post("/predict")

    assert response.status_code == 400
    assert response.json["success"] is False


def test_predict_empty_filename(client):

    data = {
        "file": (io.BytesIO(b"abc"), "")
    }

    response = client.post(
        "/predict",
        data=data,
        content_type="multipart/form-data"
    )

    assert response.status_code == 400
    assert response.json["success"] is False