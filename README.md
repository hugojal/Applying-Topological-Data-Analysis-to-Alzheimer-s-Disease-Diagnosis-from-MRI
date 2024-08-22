<!DOCTYPE html>
<html>
<head>
    <title>PDF Viewer</title>
    <script src="path/to/pdf.js"></script>
</head>
<body>
    <div id="pdf-viewer"></div>
    <script>
        pdfjsLib.getDocument('Applying_Topological_Data_Analysis_to_Alzheimer_s_Disease_Diagnosis_from_MRI.pdf').promise.then(function(pdf) {
            pdf.getPage(1).then(function(page) {
                var scale = 1.5;
                var viewport = page.getViewport({scale: scale});
                var canvas = document.createElement('canvas');
                var context = canvas.getContext('2d');
                canvas.height = viewport.height;
                canvas.width = viewport.width;
                var renderContext = {
                    canvasContext: context,
                    viewport: viewport
                };
                page.render(renderContext);
                document.getElementById('pdf-viewer').appendChild(canvas);
            });
        });
    </script>
</body>
</html>
