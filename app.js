const bodyParser = require("body-parser");
const request = require("request");
const fileUpload = require("express-fileupload");
const express = require("express");

const app = express();
const port = process.env.PORT || 4000;
const hostname = "localhost";

let spawn = require("child_process").spawn;

app.use(
  bodyParser.urlencoded({
    extended: false
  })
);

app.use(fileUpload());

app.use(bodyParser.json());

app.use(function (req, res, next) {
  res.header("Access-Control-Allow-Origin", "*");
  res.header(
    "Access-Control-Allow-Headers",
    "Origin, X-Requested-With, Content-Type, Accept"
  );
  next();
});

app.post("/predict/:minimumScore", (req, res) => {

  if (Object.keys(req.files).length == 0) {
    return res.status(400).send("No files were uploaded.");
  }

  let fileName = req.files.files.name;
  let minimumScore = req.params.minimumScore

  console.log(minimumScore);

  req.files.files.mv("./predictData/" + fileName, function (err) {
    if (err) return res.status(500).send(err);
    res.send(req.files.files.name + " uploaded!");
  });

  let process = spawn("python", [
    "./utility/emotionDetection/model.py",
    "./predictData/" + fileName,
    minimumScore
  ]);

  process.stdout.on("data", function (data) {
    console.log(data.toString());
  });
});

let path = '/Users/Gear/Desktop/emotion_detection_server/result/'

app.get("/predict", (req, res) => {
  res.sendFile(path + 'JSON/data.json');
});

app.get("/video", (req, res) => {
  res.sendFile(path + 'video/output.avi')
});

app.post("/train", (req, res) => {
  let epchoNumber = req.body.epochNumber;
  console.log(epchoNumber);
  let process = spawn("python", [
    "./utility/emotionDetection/train.py",
    epchoNumber
  ]);

  process.stdout.on("data", function (data) {
    // res.send(data);
    console.log(data.toString());
  });
  res.sendStatus(200);
});

app.post("/upload/:emotion", (req, res) => {
  if (Object.keys(req.files).length == 0) {
    return res.status(400).send("No files were uploaded.");
  }

  req.files.files.mv(
    "./utility/emotionDetection/images/" + req.params.emotion + "/" + req.files.files.name,
    function (err) {
      if (err) return res.status(500).send(err);
      res.send(req.files.files.name + " uploaded!");
    }
  );
});

const url = "https://localhost:8080/";

function curl(method, body) {
  console.log("method:" + method);
  request.post({
      url: url + method,
      headers: HEADERS,
      body: body
    },
    (err, res, body) => {
      console.log("status = " + res.statusCode);
      console.log(err);
    }
  );
}

app.listen(port, hostname, () => {
  console.log(`Server running at http://${hostname}:${port}/`);
});
