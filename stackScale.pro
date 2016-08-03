PRO stackScale

; specify directory of sr images
dir = '/Users/vscholl/Documents/melt_pond/data/sr/L7test/
scene = 'LE70660092000169-SC20160715104425/' ;make sure there is a '/' at the end 

landsatNumber = scene.CharAt(2)
dir = dir + scene
filenames = file_search(dir + '*sr_band*')

; create empty array to contain all 7 bands of surface reflectance data
stacked = []

; read in each band and append to the stacked array
foreach file, filenames $  
do begin 
  print, 'filename = ', file
  im = float(read_tiff(file))
  stacked = [ [[stacked]], [[im]] ]
  print, size(stacked)
endforeach

; multiply by scale factor to convert 16bit DN to refl
scale = 0.0001
stackedScaled = stacked * scale

; write out and open in ENVI 
basename = strmid(file,strlen(dir),24)
fnameStackedScaled = dir + basename + '_stackedScaled.tif'
write_tiff, fnameStackedScaled, stackedScaled, planarconfig=2, /float


; create header including band names and wavelength values
envi_open_file, fnameStackedScaled, r_fid=fid
print, fid
ENVI_FILE_QUERY, fid, ns=ns, nl=nl, nb=nb, data_type=data_type, descrip=descrip, bnames=bnames, sensor_type=sensor_type, wavelength_units=wavelength_units, dims=dims

; specify the band names
print, 'landsat number = ', landsatNumber
if landsatNumber eq 8 then begin
  bnames = ['Coastal', 'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2'] ;L8 OLI
  wl = [0.443000, 0.482600, 0.561300, 0.654600, 0.864600, 1.609000, 2.201000] ; band center wavelengths


endif else begin
  bnames = ['Blue', 'Green', 'Red', 'NIR', 'SWIR', 'SWIR2'] ; L5 TM, L7 ETM+
  wl = [0.49, 0.56, 0.66, 0.83, 1.67, 2.24] ; band center wavelengths


endelse
print, 'band names = ', bnames
wavelength_units = 'Micrometers'

ENVI_SETUP_HEAD, fname=fnameStackedScaled, ns=ns, nl=nl, nb=nb, data_type=data_type, offset=0, interleave=0, xstart=0, ystart=0, descrip=descrip, bnames=bnames, sensor_type=sensor_type,  wavelength_units=wavelength_units, wl=wl, /write


; create mask for landsat black background and apply it ? 


; determine the number of ice/water/pond pixels in a class image
; where unclassified pixels = 0
;ice = where(classIm eq 1, iceCount)
;water = where(classIm eq 2, waterCount)
;pond = where(classIm eq 3, pondCount)




; calculate pond fraction 



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; extra things that may be useful in the future

; reshape array from [c,r,b] to be [b,c,r]
;dims = size(stackedScaled)
;cols = dims(1)
;rows = dims(2)
;bands = dims(3)
;stackedScaledReshaped = reform(stackedScaled,[bands,cols,rows])

;;; tutorial stuff from video
;device, decomposed = 0
;window, xsize = 640, ysize=480
;!P.background = 255
;map_set, /cylindrical, color = 0, /continents, /grid, /label


end

