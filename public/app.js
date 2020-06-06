let display = document.querySelector("#display");
let runScript = document.querySelector("#run-script");

function displayWaitMessage() {
  runScript.style.background = "#444";
  runScript.style.color = "#000";
  runScript.setAttribute("disabled", "disabled");
  display.style.visibility = "visible";
}

runScript.addEventListener("click", () => {
  displayWaitMessage();
});
