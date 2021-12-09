var express = require("express");
const multer = require("multer");
var router = express.Router();
const {spawn} = require('child_process');

// GET request
// Just a test API to check if server is working properly or not
router.get("/", function (req, res) {
	res.send("API is working properly!");
});

var storageBoard = multer.diskStorage({
	destination: "./images/boards",
	filename: function (req, file, cb) {
		cb(null, file.originalname )
	}
})

var uploadBoard = multer({ storage: storageBoard }).single('file');

var storageTiles = multer.diskStorage({
	destination: "./images/tiles",
	filename: function (req, file, cb) {
		cb(null, file.originalname )
	}
})

var uploadTiles= multer({ storage: storageTiles }).single('file');

router.post("/getBoard", function (req, res) {
	uploadBoard(req, res, function (err) {
		if (err instanceof multer.MulterError) {
			console.log('Multer Error: ', err)
		} else if (err) {
			console.log(err)
		} else {
			console.log('Successful Upload')
			fileName = req.file.originalname
			var dataToSend;

			// spawn new child process to call the python script
			const python = spawn("python", ["getBoard.py", fileName]);

			// collect data from script
			python.stdout.on("data", function (data) {
				console.log("Pipe data from python script ...");
				dataToSend = data.toString();
			});

			// in close event we are sure that stream from child process is closed
			python.on("close", (code) => {
				console.log(`child process close all stdio with code ${code}`);
				res.send(dataToSend);
			});
		}
	});
});

router.post("/getWord", function (req, res) {
	uploadTiles(req, res, function (err) {
		if (err instanceof multer.MulterError) {
			console.log('Multer Error: ', err)
		} else if (err) {
			console.log(err)
		} else {
			console.log('Successful Upload')
			fileName = req.file.originalname
			var dataToSend;

			// spawn new child process to call the python script
			const python = spawn("python", ["getWord.py", fileName]);

			// collect data from script
			python.stdout.on("data", function (data) {
				console.log("Pipe data from python script ...");
				dataToSend = data.toString();
			});

			// in close event we are sure that stream from child process is closed
			python.on("close", (code) => {
				console.log(`child process close all stdio with code ${code}`);
				res.send(dataToSend);
			});
		}
	});
});

module.exports = router;
