const image_input = document.querySelector("#imageInput");
var uploaded_image = "";


image_input.addEventListener("change", function(){
    const reader = new FileReader();
    reader.addEventListener("load", ()=> {
        uploaded_image = reader.result
        document.querySelector("#imageOutput").getElementsByClassName.backgroundImage = `src(${uploaded_image})`
    })
    reader.readAsDataURL(this.files[0])
})