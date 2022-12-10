import React from "react";
import SaveIcon from "@mui/icons-material/Save";
import CancelIcon from "@mui/icons-material/Cancel";

import {
  Box,
  TableContainer,
  Button,
  Snackbar,
  Table,
  TableBody,
  TableHead,
  TableRow,
} from "@mui/material";

const EditableRow = ({
  editFormData,
  handleEditFormChange,
  handleCancelClick,
}) => {
  return (
    <tr>
      <td>
        <input
          type="text"
          required="required"
          placeholder="Enter a tweet text..."
          name="tweetText"
          value={editFormData.TweetText}
          onChange={handleEditFormChange}
        ></input>
      </td>

      <td>
        <Button type="submit" variant="contained" endIcon={<SaveIcon />}>
          Save
        </Button>
        <Button
          type="button"
          variant="contained"
          endIcon={<CancelIcon />}
          onClick={handleCancelClick}
        >
          Cancel
        </Button>
      </td>
    </tr>
  );
};

export default EditableRow;