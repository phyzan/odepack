PREFIX ?= /usr/local
INCLUDEDIR ?= $(PREFIX)/include

.PHONY: install uninstall

install:
	@echo "Installing odepack headers to $(INCLUDEDIR)/odepack..."
	install -d $(INCLUDEDIR)/odepack/ode
	install -d $(INCLUDEDIR)/odepack/ndspan
	install -d $(INCLUDEDIR)/odepack/pyode
	cp -r include/ode/* $(INCLUDEDIR)/odepack/ode/
	cp -r include/ndspan/* $(INCLUDEDIR)/odepack/ndspan/
	cp -r include/pyode/*.hpp $(INCLUDEDIR)/odepack/pyode/
	cp include/odepack.hpp $(INCLUDEDIR)/odepack/
	cp include/PyMain.hpp $(INCLUDEDIR)/odepack/
	@echo "Done. You can now use: #include <odepack/odepack.hpp>"

uninstall:
	@echo "Removing odepack headers from $(INCLUDEDIR)..."
	rm -rf $(INCLUDEDIR)/odepack
	@echo "Done."
