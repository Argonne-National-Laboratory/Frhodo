# Troe Falloff Rate — Derivation and Jacobian

Frhodo represents pressure-dependent reactions in the Troe falloff form (as implemented by Cantera's `TroeRate`). This note derives $\ln k$ and its analytic Jacobian with respect to the low- and high-pressure Arrhenius parameters and the four Troe coefficients, which the optimizer uses when fitting these reactions. The symbol-to-code-variable map is listed below.

$$ k_{0} = A*T^{n}*\exp\left( - \frac{Ea}{(R*T)} \right) $$

$$ k_{0} = A*T^{n}*\exp\left( - \frac{Ea}{(R*T)} \right) $$

$$ P_{r} = \frac{k_{0}\lbrack M\rbrack}{k_{\infty}} $$

$$ k = k_{\infty}\left( \frac{P_{r}}{1 + P_{r}} \right)F $$

$$ \log(F) = \frac{\log\left( F_{cent} \right)}{1 + f^{2}} $$

$$ F_{cent} = (1 - A)*\exp{\left( - \frac{T}{T_{3}} \right) + A*\exp\left( - \frac{T}{T_{1}} \right) + \exp\left( - \frac{T_{2}}{T} \right)} $$

$$ f = \frac{\left( \log\left( P_{r} \right) + C \right)}{N - 0.14*\left( \log\left( P_{r} \right) + C \right)} $$

$$ C = - 0.4 - 0.67*\log\left( F_{cent} \right) $$

$$ N = 0.75 - 1.27*\log\left( F_{cent} \right) $$

| Symbol | Code variable |
|--------|---------------|
| $A_{0}$ | `a0` |
| $n_{0}$ | `n0` |
| $- Ea_{0}$ | `ea0` |
| $A_{\infty}$ | `a1` |
| $n_{\infty}$ | `n1` |
| $- Ea_{\infty}$ | `ea1` |
| $A$ | `A` |
| $T_{3}$ | `T3` |
| $T_{1}$ | `T1` |
| $T_{2}$ | `T2` |

$$ P_{r} = \lbrack M\rbrack\frac{a0}{a1}\ T^{n0 - n1}\exp\left( \frac{ea0 - ea1}{(R*T)} \right) $$

$$ {ln}k = {ln}\left( \frac{\lbrack M\rbrack k_{0}}{1 + P_{r}} \right) + {ln}{(F)} $$

$$ \ln k = \ln(a1) + n1\ln{(T)} + \frac{Ea1}{(R*T)} $$

$$ \ln\left( \frac{\lbrack M\rbrack k_{0}}{1 + P_{r}} \right) = \ln\left( \lbrack M\rbrack k_{0}k_{\infty} \right) - \ln{(\lbrack M\rbrack k_{0} + k_{\infty})} $$

$$ {ln}F = \frac{\log e\log\left( F_{cent} \right)}{1 + \left( \frac{\left( \log\left( P_{r} \right) + C \right)}{N - 0.14*\left( \log\left( P_{r} \right) + C \right)} \right)^{2}} $$

$$ \frac{d\ln k}{da0} = \frac{1}{a0}\left( 1 - \frac{\lbrack M\rbrack k_{0}}{\lbrack M\rbrack k_{0} + k_{\infty}} \right) + \frac{d\ln F}{d\log P_{r}}\frac{d\log P_{r}}{da0} $$

$$ \frac{d\ln k}{dn0} = \ln T\left( \frac{k_{\infty}}{\lbrack M\rbrack k_{0} + k_{\infty}} \right) + \frac{d\ln F}{d\log P_{r}}\frac{d\log P_{r}}{dn0} $$

$$ \frac{d\ln k}{dea0} = \frac{1}{RT}\left( \frac{k_{\infty}}{\lbrack M\rbrack k_{0} + k_{\infty}} \right) + \frac{d\ln F}{d\log P_{r}}\frac{d\log P_{r}}{dea0} $$

$$ \frac{d\ln k}{da1} = \frac{1}{a1}\left( 1 - \frac{k_{\infty}}{\lbrack M\rbrack k_{0} + k_{\infty}} \right) + \frac{d\ln F}{d\log P_{r}}\frac{d\log P_{r}}{da1} $$

$$ \frac{d\ln k}{dn1} = \ln T\left( \frac{\lbrack M\rbrack k_{0}}{\lbrack M\rbrack k_{0} + k_{\infty}} \right) + \frac{d\ln F}{d\log P_{r}}\frac{d\log P_{r}}{dn1} $$

$$ \frac{d\ln k}{dea1} = \frac{1}{RT}\left( \frac{\lbrack M\rbrack k_{0}}{\lbrack M\rbrack k_{0} + k_{\infty}} \right) + \frac{d\ln F}{d\log P_{r}}\frac{d\log P_{r}}{dea1} $$

$$ \frac{d\ln k}{dA} = \frac{dF_{cent}}{dA}\left( \frac{d\ln F}{dF_{cent}} + \frac{d\ln F}{dC}\frac{dC}{dF_{cent}} + \frac{d\ln F}{dN}\frac{dN}{dF_{cent}} \right) $$

$$ \frac{d\ln k}{dT3} = \frac{dF_{cent}}{dT3}\left( \frac{d\ln F}{dF_{cent}} + \frac{d\ln F}{dC}\frac{dC}{dF_{cent}} + \frac{d\ln F}{dN}\frac{dN}{dF_{cent}} \right) $$

$$ \frac{d\ln k}{dT1} = \frac{dF_{cent}}{dT1}\left( \frac{d\ln F}{dF_{cent}} + \frac{d\ln F}{dC}\frac{dC}{dF_{cent}} + \frac{d\ln F}{dN}\frac{dN}{dF_{cent}} \right) $$

$$ \frac{d\ln k}{dT2} = \frac{dF_{cent}}{dT2}\left( \frac{d\ln F}{dF_{cent}} + \frac{d\ln F}{dC}\frac{dC}{dF_{cent}} + \frac{d\ln F}{dN}\frac{dN}{dF_{cent}} \right) $$

$$ \ln F = \frac{\log e\log\left( F_{cent} \right)}{1 + Y} $$

$$ Y = \left( \frac{\left( \log\left( P_{r} \right) + C \right)}{N - 0.14*\left( \log\left( P_{r} \right) + C \right)} \right)^{2} $$

$$ u = N - 0.14\ v,\ \ v = \log\left( P_{r} \right) + C $$

$$ \frac{d\ln F}{dF_{cent}} = \frac{1}{F_{cent}}\frac{\log e}{\ln 10}\frac{u^{4}}{\left( u^{2} + v^{2} \right)^{2}} $$

$$ \frac{d\ln F}{d\ P_{r}} = \frac{- 2\ N\ \left( \log e \right)^{2}\log\left( F_{cent} \right)}{P_{r}\left( \left( \frac{v}{u} \right)^{2} + 1 \right)}\left( \frac{v}{u^{3}} \right) = - 2\ N\left( \log e \right)^{2}\log\left( F_{cent} \right)\frac{v}{u\left( u^{2} + v^{2} \right)} $$

$$ \frac{d\ln F}{dN} = \frac{2\ \log e\log\left( F_{cent} \right)}{\left( \left( \frac{v}{u} \right)^{2} + 1 \right)^{2}}\left( \frac{v^{2}}{u^{3}} \right) = 2\ \log e\log\left( F_{cent} \right)\frac{u\ v^{2}}{\left( u^{2} + v^{2} \right)^{2}} $$

$$ \frac{d\ln F}{dC} = - \frac{2\ N\log e\log\left( F_{cent} \right)}{\left( \left( \frac{v}{u} \right)^{2} + 1 \right)^{2}}\left( \frac{\ v}{u^{3}} \right) = - 2N\log e\log\left( F_{cent} \right)\frac{u\ v}{\left( u^{2} + v^{2} \right)^{2}} $$

$$ \frac{dF_{cent}}{dA} = \exp\left( - \frac{T}{T_{1}} \right) - \exp\left( - \frac{T}{T_{3}} \right) $$

$$ \frac{dF_{cent}}{dT_{3}} = (1 - A)\frac{T}{T_{3}^{2}}\exp\left( - \frac{T}{T_{3}} \right) $$

$$ \frac{dF_{cent}}{dT_{1}} = A\frac{T}{T_{1}^{2}}\exp\left( - \frac{T}{T_{1}} \right) $$

$$ \frac{dF_{cent}}{dT_{2}} = \frac{- 1}{T}\exp\left( \frac{- T_{2}}{T} \right) $$

$$ \frac{dC}{dF_{cent}} = - \frac{0.67}{F_{cent}}\log{(e)} $$

$$ \frac{dN}{dF_{cent}} = - \frac{1.27}{F_{cent}}\log{(e)} $$

$$ \frac{d\ln k}{da0} = \frac{1}{a0}\left( 1 - \frac{\lbrack M\rbrack k_{0}}{\lbrack M\rbrack k_{0} + k_{\infty}} \right) - 2\ N\left( \log e \right)^{2}\log\left( F_{cent} \right)\frac{v}{u\left( u^{2} + v^{2} \right)}\frac{d\log P_{r}}{da0} $$

$$ \frac{d\ln k}{dn0} = \ln T\left( \frac{k_{\infty}}{\lbrack M\rbrack k_{0} + k_{\infty}} \right) - 2\ N\left( \log e \right)^{2}\log\left( F_{cent} \right)\frac{v}{u\left( u^{2} + v^{2} \right)}\frac{d\log P_{r}}{dn0} $$

$$ \frac{d\ln k}{dea0} = \frac{1}{RT}\left( \frac{k_{\infty}}{\lbrack M\rbrack k_{0} + k_{\infty}} \right) - 2\ N\left( \log e \right)^{2}\log\left( F_{cent} \right)\frac{v}{u\left( u^{2} + v^{2} \right)}\frac{d\log P_{r}}{dea0} $$

$$ \frac{d\ln k}{da1} = \frac{1}{a1}\left( 1 - \frac{k_{\infty}}{\lbrack M\rbrack k_{0} + k_{\infty}} \right) - 2\ N\left( \log e \right)^{2}\log\left( F_{cent} \right)\frac{v}{u\left( u^{2} + v^{2} \right)}\frac{d\log P_{r}}{da1} $$

$$ \frac{d\ln k}{dn1} = \ln T\left( \frac{\lbrack M\rbrack k_{0}}{\lbrack M\rbrack k_{0} + k_{\infty}} \right) - 2\ N\left( \log e \right)^{2}\log\left( F_{cent} \right)\frac{v}{u\left( u^{2} + v^{2} \right)}\frac{d\log P_{r}}{dn1} $$

$$ \frac{d\ln k}{dea1} = \frac{1}{RT}\left( \frac{\lbrack M\rbrack k_{0}}{\lbrack M\rbrack k_{0} + k_{\infty}} \right) - 2\ N\left( \log e \right)^{2}\log\left( F_{cent} \right)\frac{v}{u\left( u^{2} + v^{2} \right)}\frac{d\log P_{r}}{dea1} $$

$$ \frac{d\ln k}{dA} = \frac{dF_{cent}}{dA}\left( \frac{1}{F_{cent}}\frac{\log e}{\ln 10}\frac{u^{4}}{\left( u^{2} + v^{2} \right)^{2}} - 2N\log e\log\left( F_{cent} \right)\frac{u\ v}{\left( u^{2} + v^{2} \right)^{2}}\frac{dC}{dF_{cent}} + 2\ \log e\log\left( F_{cent} \right)\frac{u\ v^{2}}{\left( u^{2} + v^{2} \right)^{2}}\frac{dN}{dF_{cent}} \right) $$

$$ \frac{d\ln k}{dT3} = \frac{dF_{cent}}{dT3}\left( \frac{1}{F_{cent}}\frac{\log e}{\ln 10}\frac{u^{4}}{\left( u^{2} + v^{2} \right)^{2}} - 2N\log e\log\left( F_{cent} \right)\frac{u\ v}{\left( u^{2} + v^{2} \right)^{2}}\frac{dC}{dF_{cent}} + 2\ \log e\log\left( F_{cent} \right)\frac{u\ v^{2}}{\left( u^{2} + v^{2} \right)^{2}}\frac{dN}{dF_{cent}} \right) $$

$$ \frac{d\ln k}{dT1} = \frac{dF_{cent}}{dT1}\left( \frac{1}{F_{cent}}\frac{\log e}{\ln 10}\frac{u^{4}}{\left( u^{2} + v^{2} \right)^{2}} - 2N\log e\log\left( F_{cent} \right)\frac{u\ v}{\left( u^{2} + v^{2} \right)^{2}}\frac{dC}{dF_{cent}} + 2\ \log e\log\left( F_{cent} \right)\frac{u\ v^{2}}{\left( u^{2} + v^{2} \right)^{2}}\frac{dN}{dF_{cent}} \right) $$

$$ \frac{d\ln k}{dT2} = \frac{dF_{cent}}{dT2}\left( \frac{1}{F_{cent}}\frac{\log e}{\ln 10}\frac{u^{4}}{\left( u^{2} + v^{2} \right)^{2}} - 2N\log e\log\left( F_{cent} \right)\frac{u\ v}{\left( u^{2} + v^{2} \right)^{2}}\frac{dC}{dF_{cent}} + 2\ \log e\log\left( F_{cent} \right)\frac{u\ v^{2}}{\left( u^{2} + v^{2} \right)^{2}}\frac{dN}{dF_{cent}} \right) $$

$$ \frac{dC}{dF_{cent}} = - \frac{0.67}{F_{cent}}\log{(e)} $$

$$ \frac{dN}{dF_{cent}} = - \frac{1.27}{F_{cent}}\log{(e)} $$

$$ \frac{d\ln k}{da0} = \frac{1}{a0}\left( 1 - \frac{\lbrack M\rbrack k_{0}}{\lbrack M\rbrack k_{0} + k_{\infty}} \right) - 2\log\left( F_{cent} \right)N\left( \log e \right)^{3}\frac{v}{u\left( u^{2} + v^{2} \right)}\frac{1}{k_{0}}T^{n0}\exp\left( - \frac{Ea0}{RT} \right) $$

$$ \frac{d\ln k}{dn0} = \ln T\left( \frac{k_{\infty}}{\lbrack M\rbrack k_{0} + k_{\infty}} \right) - 2\log\left( F_{cent} \right)N\left( \log e \right)^{3}\frac{v}{u\left( u^{2} + v^{2} \right)}\frac{1}{k_{0}}\ln(T)a0\ T^{n0}\exp\left( - \frac{Ea0}{RT} \right) $$

$$ \frac{d\ln k}{dea0} = \frac{1}{RT}\left( \frac{k_{\infty}}{\lbrack M\rbrack k_{0} + k_{\infty}} \right) + 2\log\left( F_{cent} \right)N\left( \log e \right)^{3}\frac{v}{u\left( u^{2} + v^{2} \right)}\frac{1}{k_{0}}\frac{1}{RT}a0T^{n0}\exp\left( - \frac{Ea0}{RT} \right) $$

$$ \frac{d\ln k}{da1} = \frac{1}{a1}\left( 1 - \frac{k_{\infty}}{\lbrack M\rbrack k_{0} + k_{\infty}} \right) + 2\log\left( F_{cent} \right)N\left( \log e \right)^{3}\frac{v}{u\left( u^{2} + v^{2} \right)}\frac{1}{k_{\infty}}T^{n1}\exp\left( - \frac{Ea1}{RT} \right) $$

$$ \frac{d\ln k}{dn1} = \ln T\left( \frac{\lbrack M\rbrack k_{0}}{\lbrack M\rbrack k_{0} + k_{\infty}} \right) + 2\log\left( F_{cent} \right)N\left( \log e \right)^{3}\frac{v}{u\left( u^{2} + v^{2} \right)}\frac{1}{k_{\infty}}\ln(T)a1T^{n1}\exp\left( - \frac{Ea1}{RT} \right) $$

$$ \frac{d\ln k}{dea1} = \frac{1}{RT}\left( \frac{\lbrack M\rbrack k_{0}}{\lbrack M\rbrack k_{0} + k_{\infty}} \right) - 2\log\left( F_{cent} \right)N\left( \log e \right)^{3}\frac{v}{u\left( u^{2} + v^{2} \right)}\frac{1}{k_{\infty}}\frac{1}{RT}a1T^{n1}\exp\left( - \frac{Ea1}{RT} \right) $$

$$ \frac{d\ln k}{dA} = \left( \exp\left( - \frac{T}{T_{1}} \right) - \exp\left( - \frac{T}{T_{3}} \right) \right)\left( \frac{u\log e}{{F_{cent}\left( u^{2} + v^{2} \right)}^{2}} \right)\left( \frac{u^{3}}{\ln(10)} - 2\log\left( F_{cent} \right)v\log(e)(1.27\ v - 0.67\ N) \right) $$

$$ \frac{d\ln k}{dT3} = \left( (1 - A)\frac{T}{T_{3}^{2}}\exp\left( - \frac{T}{T_{3}} \right) \right)\left( \frac{u\log e}{{F_{cent}\left( u^{2} + v^{2} \right)}^{2}} \right)\left( \frac{u^{3}}{\ln(10)} - 2\log\left( F_{cent} \right)v\log(e)(1.27\ v - 0.67\ N) \right) $$

$$ \frac{d\ln k}{dT1} = \left( A\frac{T}{T_{1}^{2}}\exp\left( - \frac{T}{T_{1}} \right) \right)\left( \frac{u\log e}{{F_{cent}\left( u^{2} + v^{2} \right)}^{2}} \right)\left( \frac{u^{3}}{\ln(10)} - 2\log\left( F_{cent} \right)v\log(e)(1.27\ v - 0.67\ N) \right) $$

$$ \frac{d\ln k}{dT2} = \left( \frac{- 1}{T}\exp\left( \frac{- T_{2}}{T} \right) \right)\left( \frac{u\log e}{{F_{cent}\left( u^{2} + v^{2} \right)}^{2}} \right)\left( \frac{u^{3}}{\ln(10)} - 2\log\left( F_{cent} \right)v\log(e)(1.27\ v - 0.67\ N) \right) $$

$$ k = A*T^{n}*\exp\left( - \frac{Ea}{(R*T)} \right) $$

$$ \frac{dk}{dA} = T^{n}\exp\left( - \frac{Ea}{RT} \right) $$

$$ \frac{dk}{dn} = \ln(T)AT^{n}\exp\left( - \frac{Ea}{RT} \right) $$

$$ \frac{dk}{dEa} = \frac{- 1}{RT}AT^{n}\exp\left( - \frac{Ea}{RT} \right) $$

$$ \frac{d\ln k}{da0} = \frac{1}{a0}\left( 1 - \frac{\lbrack M\rbrack k_{0}}{\lbrack M\rbrack k_{0} + k_{\infty}} - 2\log\left( F_{cent} \right)N\left( \log e \right)^{3}\frac{v}{u\left( u^{2} + v^{2} \right)} \right) $$

$$ \frac{d\ln k}{dn0} = \ln T\left( \frac{k_{\infty}}{\lbrack M\rbrack k_{0} + k_{\infty}} - 2\log\left( F_{cent} \right)N\left( \log e \right)^{3}\frac{v}{u\left( u^{2} + v^{2} \right)} \right) $$

$$ \frac{d\ln k}{dea0} = \frac{1}{RT}\left( \frac{k_{\infty}}{\lbrack M\rbrack k_{0} + k_{\infty}} + 2\log\left( F_{cent} \right)N\left( \log e \right)^{3}\frac{v}{u\left( u^{2} + v^{2} \right)} \right) $$

$$ \frac{d\ln k}{da1} = \frac{1}{a1}\left( 1 - \frac{k_{\infty}}{\lbrack M\rbrack k_{0} + k_{\infty}} + 2\log\left( F_{cent} \right)N\left( \log e \right)^{3}\frac{v}{u\left( u^{2} + v^{2} \right)} \right) $$

$$ \frac{d\ln k}{dn1} = \ln T\left( \frac{\lbrack M\rbrack k_{0}}{\lbrack M\rbrack k_{0} + k_{\infty}} + 2\log\left( F_{cent} \right)N\left( \log e \right)^{3}\frac{v}{u\left( u^{2} + v^{2} \right)} \right) $$

$$ \frac{d\ln k}{dea1} = \frac{1}{RT}\left( \frac{\lbrack M\rbrack k_{0}}{\lbrack M\rbrack k_{0} + k_{\infty}} - 2\log\left( F_{cent} \right)N\left( \log e \right)^{3}\frac{v}{u\left( u^{2} + v^{2} \right)} \right) $$

$$ \frac{d\ln k}{dA} = \left( \exp\left( - \frac{T}{T_{1}} \right) - \exp\left( - \frac{T}{T_{3}} \right) \right)\left( \frac{u\log e}{{F_{cent}\left( u^{2} + v^{2} \right)}^{2}} \right)\left( \frac{u^{3}}{\ln(10)} - 2\log\left( F_{cent} \right)v\log(e)(1.27\ v - 0.67\ N) \right) $$

$$ \frac{d\ln k}{dT3} = \left( (1 - A)\frac{T}{T_{3}^{2}}\exp\left( - \frac{T}{T_{3}} \right) \right)\left( \frac{u\log e}{{F_{cent}\left( u^{2} + v^{2} \right)}^{2}} \right)\left( \frac{u^{3}}{\ln(10)} - 2\log\left( F_{cent} \right)v\log(e)(1.27\ v - 0.67\ N) \right) $$

$$ \frac{d\ln k}{dT1} = \left( A\frac{T}{T_{1}^{2}}\exp\left( - \frac{T}{T_{1}} \right) \right)\left( \frac{u\log e}{{F_{cent}\left( u^{2} + v^{2} \right)}^{2}} \right)\left( \frac{u^{3}}{\ln(10)} - 2\log\left( F_{cent} \right)v\log(e)(1.27\ v - 0.67\ N) \right) $$

$$ \frac{d\ln k}{dT2} = \left( \frac{- 1}{T}\exp\left( \frac{- T_{2}}{T} \right) \right)\left( \frac{u\log e}{{F_{cent}\left( u^{2} + v^{2} \right)}^{2}} \right)\left( \frac{u^{3}}{\ln(10)} - 2\log\left( F_{cent} \right)v\log(e)(1.27\ v - 0.67\ N) \right) $$
