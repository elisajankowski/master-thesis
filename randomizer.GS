function doGet() {
  var output = HtmlService.createHtmlOutputFromFile("Index.html");
  return output.setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL);
}