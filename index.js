require("dotenv").config();

const { spawn, exec } = require("child_process");
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
  res.render("index.ejs");
});

app.post("/", upload.single("file"), async (req, res) => {
  if (req.file) {
    const file = req.file;
    const path = file.path;
    return res.render("visualize.ejs", {
      filepath: path,
    });
  }
  res.send("File was not found");
});

app.post("/result", (req, res) => {
  let fileinput = ".\\";
  fileinput += req.body.file_path;
  let file_context = "";
  let images = [];
  let algorithms = [];

  //Check the existence of the file
  if (!fileinput) {
    return res.send("Something went wrong! Upload file again");
  }

  //Get result from python script
  /* Using spawn/exec childprocess */
  fileinput.split("\\").join("\\\\");
  console.log("input: " + fileinput);

  exec("ipython t.py " + fileinput, (error, stdout, stderr) => {
    if (error) {
      res.send(`error: ${error.message}`);
      return;
    }
    if (stderr) {
      // ignore stderr
      console.log(`stderr: ${stderr}`);
    }

    /* OR */
    // const pyScript = spawn("ipython ", ["t.py", fileinput]); //Runs the python script

    // pyScript.stderr.on("data", (data) => {
    //   console.log(`stderr: ${data}`);
    // });

    // pyScript.on("error", (error) => {
    //   console.log(`error: ${error.message}`);
    // });

    // //After script has finsihed running
    // pyScript.on("close", (code) => {
    //   console.log(`child process exited with code ${code}`);
    //After running python script
    if (stdout) {
      async function readIntoAndRetrieveFilesSync() {
        const file_name = path.join(__dirname, "file.txt");
        const imageFolder = path.join(__dirname, "public", "images");
        const algorithmFolder = path.join(__dirname, "public", "algorithms");

        file_context = fs.readFileSync(file_name, "utf8");
        if (!file_context) {
          console.log("file_context: " + err);
          file_context = "No Content in File, run again!";
        }

        //get images
        let imgFiles = fs.readdirSync(imageFolder);
        imgFiles.forEach((file) => {
          let src = path.join("images", file);
          console.log(src);
          images.push(src);
        });

        //get algorithm images
        let algoFiles = fs.readdirSync(algorithmFolder);
        algoFiles.forEach((file) => {
          let src = path.join("algorithms", file);
          console.log(src);
          algorithms.push(src);
        });

        if (images && algorithms && file_context) {
          return { images, algorithms, file_context };
        }
      }
      //call synchronous function
      readIntoAndRetrieveFilesSync()
        .then((data) => {
          const { images, algorithms, file_context } = data;
          // console.log(`Algo: ${data.algorithm}`);
          // console.log(`file_context: ${data.file_context}`);
          // console.log(`images: ${data.images}`);

          if (data) {
            res.render("result.ejs", {
              fileContext: file_context
                ? file_context
                : "Nothing here, check python script",
              images: images ? images : "Could not retrieve images",
              algorithms: algorithms ? algorithms : "Could not retrieve images",
            });
          }
        })
        .catch(() => {
          res.send("Could not retrieve images to display");
        });
    }
  });
});

app.listen(port, () => console.log(`listening at ${hostname} on port:${port}`));
