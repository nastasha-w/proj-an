/* For extracting complicated-looking selections from 2d arrays
   Intended for cilinder extraction or exclusion from a list of slices
 */
 


/*  inputs:
    centres, depths, impactparameters, numtosel: cilinder data for selection, and length of the arrays
    slice2d, xpixels, ypixels: slice of the simulation and its size
    x,y,z len, cen: centres and dimesions of the box in same units as cilinder data
    fixnumsl: use a fixed number of slices for each cilinder 
              int > 0: number of slices, centering done as best as possible
              -1:      include at least one slice, and others if z center is in the cilinder
              -2:      include slices only if center is in the cilinder
              -3:      include all slices that intersect the cilinder
              -4:      include all slices fully inside the cilinder, and at least one
              -5:      include all slices fully inside the cilinder 
    periodic: periodic boundary conditions in the x/y and z dimensions
    include:  include the data in the cilinders (1) or exclude it (0)    
    here, the projection axis is always called z. Rename axes if this is not the case in the actual projection
    
    output:
    2d mask of booleans: true if include, false if exclude
*/
void main(float *centres, float *depths, float *impactparameters, int numtosel,
          float *slice2d, int xpixels, int ypixels, 
          float xlen, float ylen, float zlen, float xcen, float ycen, float zcen,
          int fixnumsl, int periodicxy, int periocdicz, int include,
          int *mask){ 
          	
  // initialise mask
  const int initval = 1 - include; // initial value is the default: 0 if including only cilinders, 1 if excluding them
  int flatind;  
  for(flatind=0; flatind < xpixels*ypixels; flatind++){
    mask[flatind] = initval;  
  } 
  
  // apply selection
  int cilind;        
  for(cilind = 0; cilind < numtosel; cilind++){
    // do selection: check z, if relevant, apply x/y mask
    if(/* some z criterion*/){
      // loop over possibly relevant x/y coordinates, select on pixel centres (check HsmlAndProject, that bit should be similar)
          
    }  
  
  }          
          
}
          
/* get_zrange functions return a selection criterion to apply to cilinder data to determine 
   whether the cilinder data should be included or not */

void get_zrange_fixnum(){}          
