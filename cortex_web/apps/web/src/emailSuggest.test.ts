import { describe, expect, it } from "vitest";
import { suggestEmail } from "./emailSuggest";

describe("suggestEmail", () => {
  it("catches the stanfodhealthcare.org incident typo", () => {
    expect(suggestEmail("natalyastewart@stanfodhealthcare.org"))
      .toBe("natalyastewart@stanfordhealthcare.org");
  });

  it("catches classic consumer-domain typos", () => {
    expect(suggestEmail("a@gmial.com")).toBe("a@gmail.com");
    expect(suggestEmail("a@gamil.com")).toBe("a@gmail.com");
    expect(suggestEmail("a@hotmial.com")).toBe("a@hotmail.com");
    expect(suggestEmail("a@iclod.com")).toBe("a@icloud.com");
    expect(suggestEmail("a@yahou.com")).toBe("a@yahoo.com");
  });

  it("catches TLD slips", () => {
    expect(suggestEmail("a@gmail.con")).toBe("a@gmail.com");
    expect(suggestEmail("a@gmail.cmo")).toBe("a@gmail.com");
  });

  it("is quiet on exact known domains", () => {
    expect(suggestEmail("a@gmail.com")).toBeNull();
    expect(suggestEmail("a@stanfordhealthcare.org")).toBeNull();
    expect(suggestEmail("A@STANFORD.EDU")).toBeNull(); // case-insensitive
  });

  it("is quiet while the user is still typing toward a known domain", () => {
    expect(suggestEmail("a@gmail.c")).toBeNull();
    expect(suggestEmail("a@stanfordhealthcare.o")).toBeNull();
    expect(suggestEmail("a@gmai")).toBeNull(); // no dot yet
    expect(suggestEmail("a@")).toBeNull();
    expect(suggestEmail("nataly")).toBeNull(); // no @ yet
  });

  it("is quiet on unrelated real domains", () => {
    expect(suggestEmail("a@bwh.harvard.edu")).toBeNull();     // distance 3+ from list
    expect(suggestEmail("a@utwente.nl")).toBeNull();
    expect(suggestEmail("a@example.org")).toBeNull();
    expect(suggestEmail("a@web.de")).toBeNull();              // short domain, cap 1
  });

  it("is quiet on real country/branding variants of known providers", () => {
    expect(suggestEmail("a@yahoo.ca")).toBeNull();
    expect(suggestEmail("a@hotmail.ca")).toBeNull();
    expect(suggestEmail("a@live.ca")).toBeNull();
    expect(suggestEmail("a@mail.com")).toBeNull();   // real, 1 edit from gmail.com
    expect(suggestEmail("a@ymail.com")).toBeNull();  // real, 1 edit from gmail.com
  });

  it("preserves the local part verbatim", () => {
    expect(suggestEmail("First.Last+tag@gmial.com")).toBe("First.Last+tag@gmail.com");
  });
});
