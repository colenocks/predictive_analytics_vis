require("dotenv").config();
const express = require("express");
const app = express();
const path = require("path");
const bodyParser = require("body-parser");
const fs = require("fs");
const multer = require("multer");

const port = process.env.PORT || 7000;
const hostname = process.env.HOST;

//set multer storage
/* 
  Save uploaded files into resources older with multer
*/
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "resources");
  },
  filename: function (req, file, cb) {
    // the null as first argument means no error
    cb(null, file.originalname);
  },
});

const upload = multer({
  storage: storage,
});

//Set Template engine
app.set("view-engine", "ejs");
app.set("views", "views");

//Serve static files
app.use(express.static(path.join(__dirname, "public")));
app.use(bodyParser.urlencoded({ extended: false }));

app.get("/", (req, res, next) => {
  res.render("index.ejs", {
    filepath: null,
  });
});

app.post("/", upload.single("file"), async (req, res) => {
  if (req.file) {
    const file = req.file;
    const path = file.path;
    return res.render("index.ejs", {
      filepath: path,
    });
  }
  res.send("File was not found");
});

app.get("/result", (req, res) => {
  // const fileinput = req.body.file_path;

  res.render("result.ejs");
});

app.listen(port, () => console.log(`listening at ${hostname} on port:${port}`));
