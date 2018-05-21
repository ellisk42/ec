# What to compile by default?
SOURCES   := $(wildcard */*.LoG)
#SOURCES   := $(wildcard */203*.LoG)
TARGETSL  := $(patsubst %.LoG,%_l.png,$(SOURCES))
TARGETSH  := $(patsubst %.LoG,%_h.png,$(SOURCES))
TARGETSVH := $(patsubst %.LoG,%_vh.png,$(SOURCES))
BINARY    := ./geomDrawFile

.PHONY: all
all: $(TARGETSL) $(TARGETSH) $(TARGETSVH)

%_l.png: %.LoG $(BINARY) Makefile
	$(BINARY) $<
%_h.png: %.LoG $(BINARY) Makefile
	$(BINARY) $<
%_vh.png: %.LoG $(BINARY) Makefile
	$(BINARY) $<

.PHONY: clean
clean:
	rm -Rf $(TARGETSL) $(TARGETSH) $(TARGETSVH) $$(find -iname "*random*")
