//Deployment to heroku fails if dotenv is required for production
if (process.env.NODE_ENV !== "production") require("dotenv").config();

const { spawn, exec } = require("child_process");
const express = require("express");
const app = express();
const path = require("path");
const bodyParser = require("body-parser");
const fs = require("fs");
const multer = require("multer");

const port = process.env.PORT || 3000;
const hostname = process.env.HOST || "localhost";

//set multer storage
/* 
  Save uploaded files into resources older with multer
*/
const storagePath = path.join(__dirname, "public", "resources");
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, storagePath);
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
    const filename = file.filename;
    // const path = file.path;
    return res.render("visualize.ejs", {
      filepath: filename,
    });
  }
  res.send("File was not found");
});

app.post("/result", (req, res) => {
  let filepath = req.body.file_path;
  let accuracyScores = [];
  let images = [];
  let imgTitles = [];
  let algorithms = [];
  let algoTitles = [];

  //Check the existence of the file
  if (!filepath) {
    return res.send("Something went wrong! Upload file again");
  }

  //Get result from python script
  /* Using spawn/exec childprocess */
  // filepath.split("/").splice(0, 1).join("\\\\");
  console.log("input: " + filepath);

  exec(`python pyscript.py ${filepath}`, (error, stdout, stderr) => {
    if (error) {
      res.send(`error: ${error.message}. [Stop Reportin']`);
      return;
    }
    if (stderr) {
      // ignore stderr
      console.log(`stderr: ${stderr}`);
    }

    //After running python script
    if (stdout) {
      async function readIntoAndRetrieveFilesSync() {
        const file_ = path.join(__dirname, "public", "file.txt");
        const imageTitles = path.join(
          __dirname,
          "public",
          "images",
          "titles.txt"
        );
        const algorithmTitles = path.join(
          __dirname,
          "public",
          "algorithms",
          "titles.txt"
        );
        const imageFolder = path.join(__dirname, "public", "images");
        const algorithmFolder = path.join(__dirname, "public", "algorithms");

        let file = fs.readFileSync(file_, "utf8");
        if (!file) {
          console.log("file_context: " + err);
        } else {
          //split and extract file contents
          let splitContents = file.split("\n");
          splitContents.forEach((content) => {
            accuracyScores.push(content.split(":"));
          });
        }

        //get images
        let imgFiles = fs.readdirSync(imageFolder);
        imgFiles.forEach((file) => {
          let src = path.join("images", file);
          images.push(src);
        });

        //get images Titles
        let img_titles = fs.readFileSync(imageTitles, "utf8");
        if (!img_titles) {
          console.log("No Images titles: " + err);
        } else {
          //split and extract file contents
          let get_img_titles = img_titles.split("\n");
          get_img_titles.forEach((content) => {
            imgTitles.push(content);
          });
        }

        //get algorithm images
        let algoFiles = fs.readdirSync(algorithmFolder);
        algoFiles.forEach((file) => {
          let src = path.join("algorithms", file);
          algorithms.push(src);
        });

        //get algorithms Titles
        let alg_titles = fs.readFileSync(algorithmTitles, "utf8");
        if (!alg_titles) {
          console.log("No algorithms titles: " + err);
        } else {
          //split and extract file contents
          let get_alg_titles = alg_titles.split("\n");
          get_alg_titles.forEach((content) => {
            algoTitles.push(content);
          });
        }

        if (images && algorithms && accuracyScores && algoTitles && imgTitles) {
          return { images, algorithms, accuracyScores, algoTitles, imgTitles };
        }
      }

      //call synchronous function
      readIntoAndRetrieveFilesSync()
        .then((data) => {
          const {
            images,
            algorithms,
            accuracyScores,
            algoTitles,
            imgTitles,
          } = data;

          if (data) {
            console.log(
              accuracyScores,
              images,
              imgTitles,
              algorithms,
              algoTitles
            );
            res.render("result.ejs", {
              accuracyScores: accuracyScores
                ? accuracyScores
                : "Nothing was Retrieved, Please check dataset or run script again",
              imgTitles: imgTitles
                ? imgTitles
                : "Could not retrieve images titles",
              images: images ? images : "Could not retrieve images",
              algoTitles: algoTitles
                ? algoTitles
                : "Could not retrieve images titles",
              algorithms: algorithms
                ? algorithms
                : "Could not retrieve images ",
            });
          }
        })
        .catch((err) => {
          res.send("Could not retrieve images to display: " + err);
        });
    }
  });
});

app.listen(port, () => console.log(`listening at ${hostname} on port:${port}`));
