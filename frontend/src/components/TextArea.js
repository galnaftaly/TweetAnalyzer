import React, { useState } from "react";
import TextareaAutosize from "@mui/base/TextareaAutosize";

const TextArea = () => {
  return (
    <TextareaAutosize
      aria-label="minimum height"
      minRows={3}
      placeholder="Type here the text of the tweet.."
      style={{ width: 700 }}
    />
  );
};
export default TextArea;
