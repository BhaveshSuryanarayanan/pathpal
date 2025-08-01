const imageFile = document.getElementById("image")

const imagePreview = document.getElementById("input-image")


function base64ToImageData(data) {
    // Create an image element
    const img = new Image();
    img.src = data; // Create a Data URL for the image

    img.onload = async function () {
      // Once the image is loaded, draw it on the canvas
      const canvas = document.getElementById('myCanvas');
      const ctx = canvas.getContext('2d');
      
      canvas.width = img.width;
      canvas.height = img.height;

      // Draw the image onto the canvas
      ctx.drawImage(img, 0, 0);

      // Get ImageData from the canvas
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

      // Log the ImageData (pixel data)
      obj = {"data": [...imageData['data']], "height" : imageData['height'], "width":imageData['width']}
      console.log([...imageData['data']].filter(value => value === 0).length)

      const response = await fetch("http://127.0.0.1:8000/roadSeg", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(obj)
      });
      const output = await response.json();
      console.log(output)
    };
  }

let Datafile = "";


imageFile.addEventListener("change",(e)=>{

    const files = e.target.files
    const reader = new FileReader();  
    // Creates a new instance of FileReader
    let imageData ;
    reader.onload = function (e) {  
         imageData = e.target.result;
         imagePreview.src = imageData
         Datafile=imageData
        //  base64_image = imageData['image']
         array = imageData.split(';');

         base64_data = array[1].split(',')[1]

         console.log(base64_data)
         console.log(imageData)
         console.log("done")
        //  console.log(base64_image)
        base64ToImageData(Datafile);
    };
    
    reader.readAsDataURL(files[0]);

})

