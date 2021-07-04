cls
clear


import delimited "G:\Economics\Finance(Prof.Heidari-Aghajanzadeh)\Data\Speculative Beta\speclativePortfoDetails.csv", encoding(UTF-8)
 
cd"D:\Dropbox\Finance(Prof.Heidari-Aghajanzadeh)\Project\Speculative Betas\Report"

foreach v in 5 10 15 20 {
	binscatter return beta if speculative != "None" &  pnumber == `v' ,by(speculative) nq(10) legend( label(1 " Nonspeculative Stocks") label(2 "Speculative Stocks") ) msymbol(Th S) line(none) title( "`v' Portfolios for each type of stocks")ytitle("r{sub:12}") xtitle("{&beta}")
	graph export `v'Sportfo.eps,replace
	graph export `v'Sportfo.png,replace
	binscatter return beta if speculative == "None" & pnumber == `v', nq(20)   line(none) title( "`v' Portfolios stocks")ytitle("r{sub:12}") xtitle("{&beta}")
	graph export `v'portfo.eps,replace
	graph export `v'portfo.png,replace
}

