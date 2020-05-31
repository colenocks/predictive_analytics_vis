require("dotenv").config();
const express = require("express");
const app = express();
const path = require("path");
const bodyParser = require("body-parser");

const port = process.env.PORT;
const hostname = process.env.HOST;

//Set Template engine
app.set("view-engine", "ejs");
app.set("views", "views");

//Serve static files
app.use(express.static(path.join(__dirname, "public")));
app.use(bodyParser.urlencoded({ extended: false }));

app.get("/", (req, res) => {
  res.render("index.ejs");
});

app.get("/result", (req, res) => {
  res.render("result.ejs");
});

app.listen(port, () => console.log(`listening at ${hostname} on port:${port}`));
