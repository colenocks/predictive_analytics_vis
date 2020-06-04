require("dotenv").config();

const { spawn } = require("child_process");
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
  const fileinput = req.body.file_path;
  
  const file_context;
  const images = [];
  const algorithms = [];

  if (!fileinput) {
    return res.send("Something went wrong! Upload file again");
  }

  //Get result from python script
  /* Using spawn */
  const pyScript = spawn("ipython ", ["t.py", fileinput.toString()]); //Runs the python script

  pyScript.stderr.on("data", (data) => {
    console.log(`stderr: ${data}`);
  });

  pyScript.on("error", (error) => {
    console.log(`error: ${error.message}`);
  });

  // After script has finsihed running 
  pyScript.on("close", (code) => {
    console.log(`child process exited with code ${code}`);
    //After running python script
    const file_name = path.join(__dirname, 'file.txt');
    fs.readFile(file_name, "utf8", function (err, data) {
      if (err) {
       return file_context = "No Content in File, run again!"
      }
      file_context = data;
    });

    //get images 
    const imageFolder = path.join(__dirname, 'images');
    fs.readdir(imageFolder, (err, imgFiles) => {
      imgFiles.forEach(file => {
        console.log(file);
        images.push(file);
      });

    //get algorithm images
    const algorithmFolder = path.join(__dirname, 'images', 'algorithms');
    fs.readdir(algorithmFolder, (err, algoFiles) => {
      algoFiles.forEach(file => {
        algorithms.push(file);
      });

    });
  });
  /* OR */
  /* Using exec */
  /*   const { exec } = require("child_process");

  exec("ls -la", (error, stdout, stderr) => {
    if (error) {
      console.log(`error: ${error.message}`);
      return;
    }
    if (stderr) {
      console.log(`stderr: ${stderr}`);
      return;
    }
    console.log(`stdout: ${stdout}`);
  }); */
  res.render("result.ejs", {
    fileContext: file_context ? file_context : "Nothing here, check python script",
    images: images? images : '',
    algorithms: algorithms ? algorithms : ''
  });
});
});

app.listen(port, () => console.log(`listening at ${hostname} on port:${port}`));
