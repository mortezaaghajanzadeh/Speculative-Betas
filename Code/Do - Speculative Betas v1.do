clear


import delimited "G:\Dropbox\Dropbox\Project\Speculative Betas\beta.csv", encoding(UTF-8)
 
cd"G:\Dropbox\Dropbox\Project\Speculative Betas\Report"




//use "PriceLimit.dta"

gen uratio = β / σu
gen rratio = β / σr



scatter return β 



egen median = median(β )
summ median


binscatter return β , nq(50) 
graph export mygraph0.png,replace
binscatter return β , nq(50) rd(1)
graph export mygraph1.png,replace
twoway lowess return β 
graph export mygraph2.png,replace


drop median

egen median = median(rratio)
summ median


binscatter return rratio , nq(50) rd( 0.04 ) 
graph export mygraph3.png,replace
twoway lowess return rratio , adjust xline(0.04)
graph export mygraph4.png,replace

