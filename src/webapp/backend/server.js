const express = require('express');
const app = express();
const bodyParser = require('body-parser');
const cors = require('cors');
const PORT = 4000;

// routes
var APIRouter = require("./routes/API");

app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// setup API endpoints
app.use("/", APIRouter);

app.listen(PORT, function() {
    console.log("Server is running on Port: " + PORT);
});
