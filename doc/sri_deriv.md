# SRI Falloff Rate — Derivation and Jacobian

This note derives $\ln k$ for a reaction in the SRI falloff form (as implemented by Cantera's `SriRate`) and its analytic Jacobian with respect to the low- and high-pressure Arrhenius parameters and the SRI coefficients $a, b, c, d, e$, used by the optimizer.

$$ k = k_{\infty}\left( \frac{P_{r}}{1 + P_{r}} \right)F $$

$$ P_{r} = \frac{k_{0}\lbrack M\rbrack}{k_{\infty}} $$

$$ k_{\infty} = A_{\infty}T^{n_{\infty}}\exp\left( - \frac{Ea_{\infty}}{RT} \right) $$

$$ P_{r} = \lbrack M\rbrack\frac{A_{0}}{A_{\infty}}\ T^{n_{0} - n_{\infty}}\exp\left( \frac{{Ea}_{\infty} - {Ea}_{0}}{RT} \right) $$

$$ F = d\left( a\exp\left( - \frac{b}{T} \right) + \exp\left( - \frac{T}{c} \right) \right)^{\frac{1}{1 + \log^{2}P_{r}}}T^{e} $$

$$ \ln F = \ln\left( d\left( a\exp\left( - \frac{b}{T} \right) + \exp\left( - \frac{T}{c} \right) \right)^{\frac{1}{1 + \log^{2}P_{r}}}T^{e} \right) $$

$$ \ln F = \ln(d) + \frac{1}{1 + \log^{2}P_{r}}\ln{\left( a\exp\left( - \frac{b}{T} \right) + \exp\left( - \frac{T}{c} \right) \right) + e\ln T} $$

$$ \ln k = \ln k_{\infty} + \ln\left( \frac{P_{r}}{1 + P_{r}} \right) + \frac{1}{1 + \log^{2}P_{r}}\ln\left( a\exp\left( - \frac{b}{T} \right) + \exp\left( - \frac{T}{c} \right) \right) + \ln(d) + e\ln T $$

$$ u = \ln\left( a\exp\left( - \frac{b}{T} \right) + \exp\left( - \frac{T}{c} \right) \right) $$

$$ \ln k = \ln k_{\infty} + \ln\left( \frac{P_{r}}{1 + P_{r}} \right) + \frac{u}{1 + \log^{2}P_{r}} + \ln(d) + e\ln T $$

$$ \ln k = \ln\left( dk_{\infty}\frac{P_{r}}{1 + P_{r}} \right) + \frac{1}{1 + \log^{2}P_{r}}\ln\left( a\exp\left( - \frac{b}{T} \right) + \exp\left( - \frac{T}{c} \right) \right) + e\ln T $$

$$ \ln k = \ln\left( dk_{\infty}\frac{P_{r}}{1 + P_{r}} \right) + \frac{\ln\left( a\exp\left( - \frac{b}{T} \right) + \exp\left( - \frac{T}{c} \right) \right)}{1 + \left( \log e \right)^{2}\left( \ln P_{r} \right)^{2}} + e\ln T $$

$$ \ln k = \ln\left( dk_{\infty}\frac{P_{r}}{1 + P_{r}} \right) + \frac{\ln\left( a\exp\left( - \frac{b}{T} \right) + \exp\left( - \frac{T}{c} \right) \right)}{1 + \left( \log e \right)^{2}\left( \ln\left( P_{r} \right) \right)^{2}} + e\ln T $$

$$ \ln P_{r} = \ln\left( \lbrack M\rbrack\frac{A_{0}}{A_{\infty}}\ T^{n_{0} - n_{\infty}}\exp\left( \frac{{Ea}_{\infty} - {Ea}_{0}}{RT} \right) \right) $$

$$ \ln^{2}P_{r} = \left( \ln\left( \lbrack M\rbrack\frac{A_{0}}{A_{\infty}} \right) + \left( n_{0} - n_{\infty} \right)\ln T + \frac{{Ea}_{\infty} - {Ea}_{0}}{RT} \right)^{2} $$

$$ \ln k = \ln\left( d\frac{k_{0}\lbrack M\rbrack}{1 + P_{r}} \right) + \frac{\ln\left( a\exp\left( - \frac{b}{T} \right) + \exp\left( - \frac{T}{c} \right) \right)}{1 + \left( \log e \right)^{2}\left( \ln\left( P_{r} \right) \right)^{2}} + e\ln T $$

$$ \ln k = \ln\left( d\lbrack M\rbrack k_{0} \right) - \ln\left( 1 + P_{r} \right) + \frac{\ln\left( a\exp\left( - \frac{b}{T} \right) + \exp\left( - \frac{T}{c} \right) \right)}{1 + s\left( \ln\left( P_{r} \right) \right)^{2}} + e\ln T $$

$$ \ln k = \ln\left( d\lbrack M\rbrack A_{0} \right) + \left( n_{0} + e \right)\ln T - \frac{Ea_{0}}{RT} - \ln\left( 1 + P_{r} \right) + \frac{\ln\left( a\exp\left( - \frac{b}{T} \right) + \exp\left( - \frac{T}{c} \right) \right)}{1 + s\left( \ln\left( P_{r} \right) \right)^{2}} $$

$$ \ln k = \ln\left( d\lbrack M\rbrack A_{0} \right) + \left( n_{0} + e \right)\ln T - \frac{Ea_{0}}{RT} - \ln\left( 1 + P_{r} \right) + \frac{\ln\left( a\exp\left( - \frac{b}{T} \right) + \exp\left( - \frac{T}{c} \right) \right)}{1 + s\left( \ln\left( P_{r} \right) \right)^{2}} $$

$$ k = k_{\infty}\left( \frac{P_{r}}{1 + P_{r}} \right)\left( a\exp\left( - \frac{b}{T} \right) + \exp\left( - \frac{T}{c} \right) \right)^{\frac{1}{1 + \log^{2}P_{r}}}d\ T^{e} $$

$$ \ln k = \ln\left( d\frac{k_{0}\lbrack M\rbrack}{1 + P_{r}} \right) + \frac{\ln\left( a\exp\left( - \frac{b}{T} \right) + \exp\left( - \frac{T}{c} \right) \right)}{1 + \left( \log e \right)^{2}\left( \ln\left( P_{r} \right) \right)^{2}} + e\ln T $$

Laplace transform?

$$ \ln k = \ln\left( d\frac{k_{0}\lbrack M\rbrack}{1 + P_{r}} \right) + \frac{\ln\left( a\exp\left( - \frac{b}{T} \right) + \exp\left( - \frac{T}{c} \right) \right)}{1 + \log^{2}P_{r}} + e\ln T $$

$$ c = \infty,\ d = 1,e = 0 $$

$$ \ln k = \ln\left( \frac{k_{0}\lbrack M\rbrack}{1 + P_{r}} \right) + \frac{\ln\left( a\exp\left( - \frac{b}{T} \right) \right)}{1 + \log^{2}P_{r}} $$

$$ \left( 1 + \log^{2}P_{r} \right)\left( \ln k - \ln\left( \frac{k_{0}\lbrack M\rbrack}{1 + P_{r}} \right) \right) = \ln(a) - \frac{b}{T} $$

$$ x = \frac{1}{T} $$

$$ \left( 1 + \log^{2}P_{r} \right)\left( \ln k - \ln\left( \frac{k_{0}\lbrack M\rbrack}{1 + P_{r}} \right) \right) = \ln(a) - bx $$

## Jacobian

$$ \frac{d}{dP_{r}} = \frac{1}{P_{r}}\left( \frac{1}{P_{r} + 1} - \frac{2u\log\left( P_{r} \right)}{\left( 1 + {\log\left( P_{r} \right)}^{2} \right)^{2}} \right) $$

$$ \frac{d}{dA_{0}} = \frac{1}{A_{0}}\left( \frac{1}{P_{r} + 1} - \frac{2u\log\left( P_{r} \right)}{\left( 1 + {\log\left( P_{r} \right)}^{2} \right)^{2}} \right) $$

$$ \frac{d}{dn_{0}} = \ln T\left( \frac{1}{P_{r} + 1} - \frac{2u\log\left( P_{r} \right)}{\left( 1 + {\log\left( P_{r} \right)}^{2} \right)^{2}} \right) $$

$$ \frac{d}{d{Ea}_{0}} = - \frac{1}{RT}\left( \frac{1}{P_{r} + 1} - \frac{2u\log\left( P_{r} \right)}{\left( 1 + {\log\left( P_{r} \right)}^{2} \right)^{2}} \right) $$

$$ \frac{d}{dA_{\infty}} = - \frac{1}{A_{\infty}}\left( \frac{1}{P_{r} + 1} - \frac{2u\log\left( P_{r} \right)}{\left( 1 + {\log\left( P_{r} \right)}^{2} \right)^{2}} \right) $$

$$ \frac{d}{dn_{\infty}} = - \ln T\left( \frac{1}{P_{r} + 1} - \frac{2u\log\left( P_{r} \right)}{\left( 1 + {\log\left( P_{r} \right)}^{2} \right)^{2}} \right) $$

$$ \frac{d}{d{Ea}_{\infty}} = \frac{1}{RT}\left( \frac{1}{P_{r} + 1} - \frac{2u\log\left( P_{r} \right)}{\left( 1 + {\log\left( P_{r} \right)}^{2} \right)^{2}} \right) $$

$$ \frac{d}{da} = \exp\left( - \frac{b}{T} \right)\left( \frac{1}{1 + \left( \log P_{r} \right)^{2}}\frac{1}{a\exp\left( - \frac{b}{T} \right) + \exp\left( - \frac{T}{c} \right)} \right) $$

$$ \frac{d}{db} = - \frac{a}{T}\exp\left( - \frac{b}{T} \right)\left( \frac{1}{1 + \left( \log P_{r} \right)^{2}}\frac{1}{a\exp\left( - \frac{b}{T} \right) + \exp\left( - \frac{T}{c} \right)} \right) $$

$$ \frac{d}{dc} = \frac{T}{c^{2}}\exp\left( - \frac{T}{c} \right)\left( \frac{1}{1 + \left( \log P_{r} \right)^{2}}\frac{1}{a\exp\left( - \frac{b}{T} \right) + \exp\left( - \frac{T}{c} \right)} \right) $$

$$ \frac{d}{dd} = \frac{1}{d} $$

$$ \frac{d}{de} = \ln T $$

Convert $\frac{2u\log\left( P_{r} \right)}{\left( 1 + {\log\left( P_{r} \right)}^{2} \right)^{2}}$ to ln $\frac{2u\log^{2}e\ln\left( P_{r} \right)}{\left( 1 + \log^{2}e{\ln\left( P_{r} \right)}^{2} \right)^{2}}$

## Hessian

$$ \begin{bmatrix} \frac{\partial\ln k}{\partial A_{0}^{2}} & \frac{\partial\ln k}{\partial A_{0}n_{0}} & \frac{\partial\ln k}{\partial A_{0}Ea_{0}} & \frac{\partial\ln k}{\partial A_{0}A_{\infty}} & \frac{\partial\ln k}{\partial A_{0}n_{\infty}} & \frac{\partial\ln k}{\partial A_{0}Ea_{\infty}} & \frac{\partial\ln k}{\partial A_{0}a} & \frac{\partial\ln k}{\partial A_{0}b} & \frac{\partial\ln k}{\partial A_{0}c} & \frac{\partial\ln k}{\partial A_{0}d} & \frac{\partial\ln k}{\partial A_{0}e} \\ \frac{\partial\ln k}{\partial n_{0}A_{0}} & \frac{\partial\ln k}{\partial n_{0}^{2}} & & & & & & & & & \\ \frac{\partial\ln k}{\partial{Ea}_{0}A_{0}} & & \frac{\partial\ln k}{\partial{Ea}_{0}^{2}} & & & & & & & & \\ \frac{\partial\ln k}{\partial A_{\infty}A_{0}} & & & \frac{\partial\ln k}{\partial A_{\infty}^{2}} & & & & & & & \\ \frac{\partial\ln k}{\partial n_{\infty}A_{0}} & & & & \frac{\partial\ln k}{\partial n_{\infty}^{2}} & & & & & & \\ \frac{\partial\ln k}{\partial{Ea}_{\infty}A_{0}} & & & & & \frac{\partial\ln k}{\partial{Ea}_{\infty}^{2}} & & & & & \\ \frac{\partial\ln k}{\partial aA_{0}} & & & & & & \frac{\partial\ln k}{\partial a^{2}} & & & & \\ \frac{\partial\ln k}{\partial bA_{0}} & & & & & & & \frac{\partial\ln k}{\partial b^{2}} & & & \\ \frac{\partial\ln k}{\partial cA_{0}} & & & & & & & & \frac{\partial\ln k}{\partial c^{2}} & & \\ \frac{\partial\ln k}{\partial dA_{0}} & & & & & & & & & \frac{\partial\ln k}{\partial d^{2}} & \\ \frac{\partial\ln k}{\partial eA_{0}} & & & & & & & & & & \frac{\partial\ln k}{\partial e^{2}} \end{bmatrix} $$

## Constraint (unused)

$$ a\exp\left( - \frac{b}{T} \right) + \exp\left( - \frac{T}{c} \right) > 0 $$

$$ a\exp\left( - \frac{b}{T} \right) > - \exp\left( - \frac{T}{c} \right) $$

$$ \ln\left( a\exp\left( - \frac{b}{T} \right) \right) < \ln\left( \exp\left( - \frac{T}{c} \right) \right) $$

$$ \ln(a) - \frac{b}{T} < - \frac{T}{c} $$

$$ \ln(a) < \frac{b}{T} - \frac{T}{c} $$

$$ a < \exp\left( \frac{b}{T} - \frac{T}{c} \right) $$

$$ \lbrack M\rbrack\frac{A_{0}}{A_{\infty}}\ T^{n_{0} - n_{\infty}}\exp\left( \frac{{Ea}_{\infty} - {Ea}_{0}}{RT} \right) $$
