PRO stack_scale_nc

dir = '/Users/vscholl/Documents/melt_pond/data/seadas/path80row8/'
filenames = file_search(dir + '*Rrs_band*')

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



end

