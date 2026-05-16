import io


def test_home(client):
    response = client.get("/")
    assert response.status_code == 200


def test_predict_no_file(client):
    response = client.post("/predict")
    assert response.status_code == 400


def test_predict_empty_filename(client):
    data = {"file": (io.BytesIO(b"test"), "")}
    response = client.post(
        "/predict",
        data=data,
        content_type="multipart/form-data"
    )
    assert response.status_code == 400


def test_predict_invalid_pipeline(client, mocker):
    mocker.patch(
        "backend.app.run_pipeline",
        return_value="NOT_DICT"
    )

    data = {"file": (io.BytesIO(b"img"), "cow.jpg")}
    response = client.post(
        "/predict",
        data=data,
        content_type="multipart/form-data"
    )

    assert response.status_code == 500


def test_predict_exception(client, mocker):
    mocker.patch(
        "backend.app.run_pipeline",
        side_effect=Exception("fail")
    )

    data = {"file": (io.BytesIO(b"img"), "cow.jpg")}
    response = client.post(
        "/predict",
        data=data,
        content_type="multipart/form-data"
    )

    assert response.status_code == 500


def test_predict_success(client, mocker):
    mocker.patch(
        "backend.app.run_pipeline",
        return_value={"success": True, "pred": 123}
    )

    data = {"file": (io.BytesIO(b"img"), "cow.jpg")}
    response = client.post(
        "/predict",
        data=data,
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert response.json["success"] is True