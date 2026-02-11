PREFIX ?= /usr/local
INCLUDEDIR ?= $(PREFIX)/include

.PHONY: install uninstall

install:
	@echo "Installing odepack headers to $(INCLUDEDIR)/odepack..."
	install -d $(INCLUDEDIR)/odepack/
	cp -r include/* $(INCLUDEDIR)/odepack/
	@echo "Done. You can now use: #include <odepack/odepack.hpp>"

uninstall:
	@echo "Removing odepack headers from $(INCLUDEDIR)..."
	rm -rf $(INCLUDEDIR)/odepack
	@echo "Done."
