$(document).ready(function(){console.log("ready!");});function delayed_math_parsing(){console.log('parsing math formulas');var i;var spans=document.querySelectorAll("span.tex");for(i=0;i<spans.length;i++){spans[i].parentNode.replaceChild(TeXZilla.toMathML(spans[i].textContent),spans[i]);}
var divs=document.querySelectorAll("div.tex");for(i=0;i<divs.length;i++){divs[i].parentNode.replaceChild(TeXZilla.toMathML(divs[i].textContent,true),divs[i]);}};(function(){window.addEventListener("DOMContentLoaded",function(){console.log('DOMContentLoaded');setTimeout(delayed_math_parsing,2000);});})();