;;; Non-interactively save all selected regions as separate files
(define (script-fu-export-crater-selected-regions image drawable)
  ;; Start
  (gimp-image-undo-group-start image)

  ;; If there are selections
  (when (= 0 (car (gimp-selection-is-empty image)))
    (let ((number 1) (prefix "") (suffix ""))
      ;; Construct filename components
      (let* ((parts (strbreakup (car (gimp-image-get-filename image)) "."))
             (coextension (unbreakupstr (reverse (cdr (reverse parts))) "."))
             )
        (set! prefix (string-append coextension "_crater_selection-" ))
        (set! suffix (string-append ".jpg")))

      ;; Convert all selections to a single path
      (plug-in-sel2path RUN-NONINTERACTIVE image drawable)

      ;; For each stroke in the path
      (let ((vectors (vector-ref (cadr (gimp-image-get-vectors image)) 0)))
        (for-each (lambda (stroke)
                    ;; Convert the stroke back into a selection
                    (let ((buffer (car (gimp-vectors-new image "buffer")))
                          (points (gimp-vectors-stroke-get-points vectors stroke)))
                      (gimp-image-insert-vectors image buffer 0 -1)
                      (apply gimp-vectors-stroke-new-from-points buffer points)
                      (gimp-vectors-to-selection buffer 2 TRUE FALSE 0 0)
                      (gimp-image-remove-vectors image buffer))

                    ;; Replace the selection with its bounding box
                    (apply (lambda (x0 y0 x1 y1)
                             (gimp-image-select-rectangle image 2 x0 y0 (- x1 x0) (- y1 y0)))
                             (cdr (gimp-selection-bounds image)))

                    ;; Extract and save the contents as a new file
                    (gimp-edit-copy drawable)
                    (let* ((image    (car (gimp-edit-paste-as-new)))
                           (drawable (car (gimp-image-get-active-layer image)))
                           (filename ""))
                      (while (or (equal? "" filename) (file-exists? filename))
                        (let* ((digits (number->string number))
                               (zeros (substring "0000" (string-length digits))))
                          (set! filename (string-append prefix zeros digits suffix)))
                        (set! number (+ number 1)))
                      (gimp-file-save RUN-NONINTERACTIVE image drawable filename filename)
                      (gimp-image-delete image)))
                  (vector->list (cadr (gimp-vectors-get-strokes vectors))))
        (gimp-image-remove-vectors image vectors))))

  ;; End
  (gimp-selection-none image)
  (gimp-image-undo-group-end image))

(script-fu-register "script-fu-export-crater-selected-regions"
                    "Export Crater Selected Regions"
                    "Export each selected region to a separate file."
                    "Andrew Kvalheim <Andrew@Kvalhe.im>"
                    "Andrew Kvalheim <Andrew@Kvalhe.im>"
                    "2012"
                    "RGB* GRAY* INDEXED*"
                    SF-IMAGE "Image" 0
                    SF-DRAWABLE "Drawable" 0)
(script-fu-menu-register "script-fu-export-crater-selected-regions" "<Image>/Select")
