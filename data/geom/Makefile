# What to compile by default?
SOURCES   := $(wildcard */*.LoG)
#SOURCES   := $(wildcard */203*.LoG)
TARGETSL  := $(patsubst %.LoG,%_l.png,$(SOURCES))
TARGETSH  := $(patsubst %.LoG,%_h.png,$(SOURCES))
TARGETSVH := $(patsubst %.LoG,%_vh.png,$(SOURCES))
BINARY    := ./geomDrawFile

.PHONY: all
all: clean $(TARGETSL) $(TARGETSH) $(TARGETSVH) others

%_l.png: %.LoG $(BINARY) Makefile
	$(BINARY) $<
%_h.png: %.LoG $(BINARY) Makefile
	$(BINARY) $<
%_vh.png: %.LoG $(BINARY) Makefile
	$(BINARY) $<

others: clean
	#cd spiral && ./generate.sh
	#cd repeat && ./generate.sh
	cd logoTasks && ./generate.sh

.PHONY: clean
clean:
	rm -Rf $$(find -iname "*.png")
	rm -Rf $$(find -iname "*.cost")
