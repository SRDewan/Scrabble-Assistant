import React, { useState } from "react";
import axios from "axios";
import "./App.css";

export default function App() {
	const [answer, setAnswer] = useState("");
	const [word, setWord] = useState("");
	const [image, setImage] = useState({ preview: '', data: '' })
	const [image2, setImage2] = useState({ preview: '', data: '' })

	const handleFileChange = (e, num) => {
		if (!num) {
			const img = {
				preview: URL.createObjectURL(e.target.files[0]),
				data: e.target.files[0],
			}
			setImage(img)
		}

		else {
			const img2 = {
				preview: URL.createObjectURL(e.target.files[0]),
				data: e.target.files[0],
			}
			setImage2(img2)
		}
	}

	function handleSubmit(num) {
		if (!num) {
			if (!image.data) {
				alert("Please upload an image of the board");
			}

			else {
				let formData = new FormData();
				formData.append('file', image.data);

				axios
					.post("http://localhost:4000/getBoard", formData)
					.then((res) => {
						setAnswer(res.data);
					})
					.catch((err) => {
						console.log(err.response);
					});
			}
		}

		else {
			if (!image2.data) {
				alert("Please upload an image of the tiles");
			}

			else {
				let formData = new FormData();
				formData.append('file', image2.data);

				axios
					.post("http://localhost:4000/getWord", formData)
					.then((res) => {
						setWord(res.data);
					})
					.catch((err) => {
						console.log(err.response);
					});
			}
		}
	}

	return (
		<div className="mainContainer">
		<div className="dataContainer">
		<div className="header">
		<span role="img" aria-label="information">
		👋
		</span>{" "}
		Hey there!
		</div>

		<div className="bio">Welcome to Scrabble Assistant</div>

		<div className="bio">
		<form>
		<div className="instruct">Upload Scrabble board image below.</div>
		{image.preview && <img src={image.preview} width='100' height='100' />}
		<hr></hr>
		<input type='file' name='file' onChange={(e) => handleFileChange(e, 0)}></input>
		</form>
		</div>

		<button className="speedButton" onClick={() => handleSubmit(0)}>
		Generate Board
		</button>

		<div className="bio">{answer}</div>

		<div className="bio">
		<form>
		<div className="instruct">Upload tiles image below.</div>
		{image2.preview && <img src={image2.preview} width='100' height='100' />}
		<hr></hr>
		<input type='file' name='file' onChange={(e) => handleFileChange(e, 1)}></input>
		</form>
		</div>

		<button className="speedButton" onClick={() => handleSubmit(1)}>
		Generate Best Word 
		</button>

		<div className="bio">{word}</div>

		</div>
		</div>
	);
}
