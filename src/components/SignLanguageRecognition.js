import React, { useRef, useState } from 'react';
import axios from 'axios';

const SignLanguageRecognition = () => {
    const [videoInput, setVideoInput] = useState(null);
    const [prediction, setPrediction] = useState('');
    const [isSpeaking, setIsSpeaking] = useState(false);

    const videoRef = useRef(null);
    const speechSynthesisRef = useRef(window.speechSynthesis);
    const utteranceRef = useRef(null);

    const handleVideoChange = (event) => {
        const file = event.target.files[0];
        setVideoInput(file);
        console.log("Video file selected: ", file);
    };

    const handlePredict = async () => {
        try {
            const formData = new FormData();
            if (videoInput instanceof Blob) {
                formData.append('video', videoInput, 'video.webm');
                console.log("Video blob appended to formData");
            } else {
                alert('Please provide a video input');
                return;
            }

            console.log("Sending request to backend...");
            const response = await axios.post('http://localhost:5000/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            console.log("Response received: ", response.data);
            setPrediction(response.data.prediction);
            speakText(response.data.prediction);
        } catch (error) {
            console.error('Error predicting:', error);
        }
    };

    const speakText = (text) => {
        utteranceRef.current = new SpeechSynthesisUtterance(text);
        speechSynthesisRef.current.speak(utteranceRef.current);
        setIsSpeaking(true);
    };

    const stopSpeaking = () => {
        if (isSpeaking) {
            speechSynthesisRef.current.cancel();
            setIsSpeaking(false);
        }
    };

    return (
        <div className="container mt-5">
            <h2 className="display-4 mb-4">Sign Language Recognition</h2>
            <div className="mb-3">
                <label>Upload Video:</label>
                <input type="file" accept="video/*" onChange={handleVideoChange} />
            </div>
            <div className="mb-3">
                <button className="btn btn-success" onClick={handlePredict}>
                    Predict
                </button>
            </div>
            {prediction && (
                <div className="mt-3">
                    <p>Prediction: {prediction}</p>
                    <button
                        className={`btn ${isSpeaking ? 'btn-danger' : 'btn-success'}`}
                        onClick={() => {
                            if (isSpeaking) {
                                stopSpeaking();
                            } else {
                                speakText(prediction);
                            }
                        }}
                    >
                        {isSpeaking ? 'Stop Speaking' : 'Speak'}
                    </button>
                </div>
            )}
        </div>
    );
};

export default SignLanguageRecognition;
