from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import pandas


class UserCheckFits(object):
    # Store fits so that we can use it in event callback
    fits_for_user_check = pandas.DataFrame()

    def __init__(self, filename, micron_per_pixel):
        """
        Let user manually check fits, removing them by clicking
        :return:
        """
        UserCheckFits.fits_for_user_check = pandas.DataFrame.from_csv(filename)

        # Set scale in pixels
        UserCheckFits.fits_for_user_check /= micron_per_pixel

        UserCheckFits.fits_for_user_check['remove'] = False

    @staticmethod
    def user_check_fits(image):
        """

        :param image:
        :return:
        """
        plot.plot_fits_for_user_confirmation(UserCheckFits.fits_for_user_check, image, UserCheckFits.on_pick)

        mask = (UserCheckFits.fits_for_user_check['remove'] == False)
        UserCheckFits.fits_for_user_check = UserCheckFits.fits_for_user_check[mask]

        UserCheckFits.fits_for_user_check.drop('remove', axis=1, inplace=True)

        # Update indices
        UserCheckFits.fits_for_user_check.reset_index(inplace=True)

        return UserCheckFits.fits_for_user_check

    @classmethod
    def on_pick(cls, event):
        """
        User clicked on a fit
        :param event:
        """
        # Get index from label
        fit_number = int(event.artist.get_label())

        if not UserCheckFits.fits_for_user_check['remove'][fit_number]:
            event.artist.set_edgecolor('r')
            plot.set_annotation_color(fit_number, 'r')
            UserCheckFits.fits_for_user_check.set_value(fit_number, 'remove', True)
        else:
            event.artist.set_edgecolor('b')
            plot.set_annotation_color(fit_number, 'b')
            UserCheckFits.fits_for_user_check.set_value(fit_number, 'remove', False)

        event.canvas.draw()
