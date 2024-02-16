var localVideo = document.getElementById('local_video');
var localCanvas = document.getElementById('local_canvas');
var localImgDep = document.getElementById('local_img_depth');
var localImgObj = document.getElementById('local_img_object');
var localSave = document.getElementById('local_save_checkbox');
var localRecord = document.getElementById('local_record_checkbox');
// var localSound = document.getElementById('local_out_sound');
var localStream = null;
// var voiceInterval = Date.now();
// var outputName = '';
// var nextTime = 100;
var soundSecond = '';

// --- prefix -----
navigator.getUserMedia  = navigator.getUserMedia    || navigator.webkitGetUserMedia ||
                        navigator.mozGetUserMedia || navigator.msGetUserMedia;

// ---------------------- video handling ----------------------- 
// start local video
function startVideo() {
    navigator.mediaDevices.getUserMedia({video: { width: 320, height: 320, facingMode: "environment" }, audio: false})   /* 1:1 */
    .then(function (stream) { // success
        localStream = stream;
        playVideo(localVideo, stream);
    }).catch(function (error) { // error
        console.error('getUserMedia error:', error);
        return;
    });
}

// stop local video
function stopVideo() {
    pauseVideo(localVideo);
    stopLocalStream(localStream);
}

function stopLocalStream(stream) {
    let tracks = stream.getTracks();
    if (! tracks) {
        console.warn('NO tracks');
        return;
    }

    for (let track of tracks) {
        track.stop();
    }
}

function playVideo(element, stream) {
    if ('srcObject' in element) {
        element.srcObject = stream;
    }
    else {
        element.src = window.URL.createObjectURL(stream);
    }
    element.play();
    element.volume = 0;
}

function pauseVideo(element) {
    element.pause();
    if ('srcObject' in element) {
        element.srcObject = null;
    }
    else {
        if (element.src && (element.src !== '') ) {
        window.URL.revokeObjectURL(element.src);
        }
        element.src = '';
    }
}

function send_img(){
    var fData = new FormData();
    
    //canvasを取得
    var context = localCanvas.getContext('2d').drawImage(localVideo, 0, 0, localCanvas.width, localCanvas.height);
    var base64 = localCanvas.toDataURL('image/jpeg');
    fData.append('img', base64);
    fData.append('before_second', soundSecond);

    var flag;
    if (localSave.checked) {
        flag = true;
        // console.log("true");
        fData.append('save', flag);
    } else {
        flag = false;
        // console.log("false");
        fData.append('save', flag);
    }
    
    if (localRecord.checked) {
        // console.log("click")
        flag = true;
        fData.append('record', flag);
    } else {
        // console.log("not click")
        flag = false;
        fData.append('record', flag);
    }


    // ajax送信
    $.ajax({
        //画像処理サーバーに返す場合
        // url: 'http://localhost:5000',
        url: 'https://t7w9gc6r-5000.asse.devtunnels.ms/',
        type: 'POST',
        traditional: true,
        data: fData ,
        contentType: false,
        processData: false,
        success: function(data, dataType) {
            //非同期で通信成功時に読み出される [200 OK 時]
            // console.log('Success', data.sound);
            localImgDep.src = data.depth_result;
            localImgObj.src = data.object_result;
            soundSecond = data.second;

            // 発言を作成
            if (data.sound != null) {
                // let now = Date.now();
                // if (Math.abs(now - voiceInterval) > 4000) {
                    // const uttr = new SpeechSynthesisUtterance(data.sound);
                    // 発言を再生 (発言キューに発言を追加)
                    // speechSynthesis.speak(uttr);
                    // console.log('sub : ' + (now - voiceInterval));
                    // voiceInterval = now;
                const uttr = new SpeechSynthesisUtterance(data.sound);
                uttr.lang = 'ja-JP';
                speechSynthesis.speak(uttr);
                // }
            }
        },
        error: function(XMLHttpRequest, textStatus, errorThrown) {
            //非同期で通信失敗時に読み出される
            console.log('Error : ' + errorThrown);
        }
    });
}

window.onload = function() {
    startVideo();
}

// 16.6, 62.5, 100, 66.6
setTimeout( function() {
    const spinner = document.getElementById('loading');
    spinner.classList.add('loaded');
    const intervalId = setInterval(() => {
        // console.log(nextTime);
        send_img();
    }, 200);
}, 5000);