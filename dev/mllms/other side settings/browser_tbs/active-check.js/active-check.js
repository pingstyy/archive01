setTimeout(function() {
    if (document.hidden) {
        // Page is currently hidden
        console.log("Hidden");
    } else {
        // Page is currently visible
        console.log("Not hidden");
    }
}, 2000);
