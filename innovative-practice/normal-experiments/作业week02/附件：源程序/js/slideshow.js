var number = 1;
function fun(){
	number++;
	if(number > 3){
		number = 1;
	}
	var elite = document.getElementById("elite");
	elite.src = "images/elite"+number+".jpg";
}
setInterval(fun,3000);