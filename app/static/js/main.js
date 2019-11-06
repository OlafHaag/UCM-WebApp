// A $( document ).ready() block.
$( document ).ready(function() {
    console.log( "ready!" );
});


/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */
function delayed_math_parsing() {
  console.log('parsing math formulas');
  var i;
  var spans = document.querySelectorAll("span.tex");
  for (i = 0; i < spans.length; i++) {
    spans[i].parentNode.
      replaceChild(TeXZilla.toMathML(spans[i].textContent), spans[i]);
  }
  var divs = document.querySelectorAll("div.tex");
  for (i = 0; i < divs.length; i++) {
    divs[i].parentNode.
      replaceChild(TeXZilla.toMathML(divs[i].textContent, true), divs[i]);
  }
};

(function() {
  window.addEventListener("DOMContentLoaded",  function() {
    setTimeout(delayed_math_parsing, 2000);
  });
})();
